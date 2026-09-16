#!/usr/bin/env python3
# ===========================================================================
#  Merge a LoRA adapter into its base model, and talk to the result.
#
#      python scripts/merge.py --adapter runs/<run-id>/final_adapter --out ./merged
#      python scripts/merge.py --adapter runs/<run-id>/final_adapter --ask "Who are you?"
#      python scripts/merge.py --merged ./merged --chat
#
#  Standalone on purpose. It needs no config, no run bundle and no pipeline: an
#  adapter directory and a base model are the whole input, so it works on an
#  adapter someone sent you, one pulled out of S3, or one from a run this
#  checkout knows nothing about.
#
#  WHAT MERGING IS
#  ---------------
#  A LoRA adapter is not a model. It is a set of low-rank deltas that have to be
#  added into a base model's weights before anything can run - which `peft` does
#  in memory every time you load one. Merging does that addition once and writes
#  the result, so what comes out is an ordinary checkpoint that
#  `AutoModelForCausalLM.from_pretrained` loads with no peft installed at all.
#
#  Bigger on disk (the whole model rather than the deltas) and no longer
#  swappable, but self-contained - which is what you want for serving, for
#  handing to someone else, or for anything that does not know what a LoRA is.
#
#  THE VOCABULARY TRAP
#  -------------------
#  Stock Qwen checkpoints pad `vocab_size` up to a multiple of 128 for tensor
#  alignment: 151936 against a tokenizer with 151665 real tokens. A checkpoint
#  fine-tuned through `resize_token_embeddings(len(tokenizer))` has had that
#  padding trimmed - and kd.train trims the student to match such a teacher
#  before training, so the adapter is built against the trimmed width.
#
#  Hand that adapter to an untrimmed base and peft refuses the state dict on a
#  shape mismatch. So the base is trimmed first, to the same width, worked out
#  from whichever of these is available:
#
#      --vocab-size N                     you said so
#      the adapter's kd-meta.json         what kd.train recorded beside it
#      the adapter's own embedding        older adapters saved a resized copy
#      the run bundle beside the adapter  config.resolved.yaml names the teacher
#
#  and left alone when none of them apply, which is the ordinary case.
#
#  That logic is NOT in this file. It lives in kd.merge, with the merge itself,
#  because `kd arena` and `kd publish` have to get past exactly the same trap
#  and three copies of the answer is three chances to have a different one. This
#  script is the by-hand front end to it.
# ===========================================================================

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def log(message=""):
    print(message, file=sys.stderr)


class _Log:
    """The .info surface kd.remote.s3 and kd.merge expect, wired to this script's
    stderr - so their progress lands with this script's, not in stdout beside
    the model's answer."""

    def info(self, message):
        log(str(message))


def localise(where, label):
    """An s3:// URI fetched into the shared cache; anything else unchanged.

    Thin wrapper over kd.paths.localise so this script and `kd arena` cannot
    disagree about where a fetched adapter lands - they share the cache, so an
    adapter one of them pulled down is already there for the other.
    """
    from kd import paths

    try:
        return paths.localise(where, log=_Log(), label=label)
    except RuntimeError as exc:
        raise SystemExit(f"xx  {exc}") from exc


def adapter_base(adapter_dir):
    """The base model the adapter was trained against, from its own config."""
    path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(path):
        raise SystemExit(
            f"xx  {adapter_dir} has no adapter_config.json, so it is not a PEFT "
            f"adapter.\n"
            f"    If it is an MLX or unsloth adapter, convert it first:\n"
            f"      kd convert-adapter --adapter {adapter_dir} --base <base> "
            f"--out ./peft-adapter")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("base_model_name_or_path")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
DTYPES = {"auto": None, "float32": "float32", "bfloat16": "bfloat16",
          "float16": "float16"}


def pick_device(requested):
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name, device):
    import torch

    if name != "auto":
        return getattr(torch, name)
    # float32 everywhere but CUDA: it is correct on every backend, and bfloat16
    # on CPU is slow and often unstable.
    return torch.bfloat16 if device == "cuda" else torch.float32


def merge(adapter_dir, base_id=None, vocab=None, device="auto", dtype="auto"):
    """Base + adapter, added together, as one ordinary model.

    The addition itself, and the vocabulary trap it has to get past, live in
    `kd.merge` - the same code `kd arena` scores with and `kd publish` ships. A
    model merged by hand here is therefore the model they produce, which is the
    only way "it worked when I tried it" means anything.
    """
    import torch  # noqa: F401 - imported for the side effect of a clear error
    from transformers import AutoTokenizer

    from kd import merge as kd_merge

    base_id = base_id or adapter_base(adapter_dir)
    if not base_id:
        raise SystemExit(
            "xx  the adapter does not record its base model, so --base is required")

    device = pick_device(device)
    torch_dtype = resolve_dtype(dtype, device)
    log(f"==> base    : {base_id}")
    log(f"==> adapter : {adapter_dir}")
    log(f"==> device  : {device} ({str(torch_dtype).replace('torch.', '')})")

    # From the adapter directory when it has one: kd.train saves the tokenizer
    # beside the weights precisely so a merged model cannot end up paired with a
    # different one than it was trained against. Read before the merge because
    # its length is one of the things the width routes need.
    source_dir = adapter_dir if os.path.isfile(
        os.path.join(adapter_dir, "tokenizer_config.json")) else base_id
    log(f"==> tokenizer: {source_dir}")
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = kd_merge.merge_adapter(
        adapter_dir, base_id, tokenizer_length=len(tokenizer), vocab=vocab,
        dtype=torch_dtype, device=device, log=_Log())
    return model.eval(), tokenizer, device


def load_merged(path, device="auto", dtype="auto"):
    """An already-merged model, loaded as an ordinary checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    torch_dtype = resolve_dtype(dtype, device)
    log(f"==> merged model: {path}  on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch_dtype, low_cpu_mem_usage=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


# --------------------------------------------------------------------------- #
# Talking to it
# --------------------------------------------------------------------------- #
def generate(model, tokenizer, prompt, device, max_new_tokens=512,
             temperature=0.0, system=None, stream=False, think=False):
    """Answer `prompt`. With stream=True the answer is printed as it is decoded.

    think=True leaves the model's reasoning block open, so on a Qwen3-style
    model the deliberation is generated and shown ahead of the answer, tags
    included. Off by default: that trace can eat the whole token budget before
    an answer appears, and every other reader of this model - training, the
    confidence probe, the arena - runs with it closed.

    A silent model.generate() is indistinguishable from a hung one - on CPU a
    512-token answer is minutes of nothing. Streaming makes the wait legible,
    and the token count printed first says how much of the prompt actually
    arrived, which is the number you want when a paste looks truncated.
    """
    import torch

    from kd.teacher import render_prompt

    # The same rendering training used, which for a model that opens with a
    # reasoning block positions the prompt past it - unless asked not to.
    text = render_prompt(tokenizer, prompt, system=system, enable_thinking=think)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    greedy = temperature <= 0

    streamer = None
    if stream:
        from transformers import TextStreamer
        log(f"[{inputs.input_ids.shape[1]} prompt tokens in, generating up to "
            f"{max_new_tokens} - Ctrl-C to cut it short]")
        # <think>/</think> are ordinary tokens on Qwen3, so they survive this
        # and a --think trace arrives with its boundaries intact.
        streamer = TextStreamer(tokenizer, skip_prompt=True,
                                skip_special_tokens=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            top_p=None if greedy else 0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            streamer=streamer,
        )
    completion = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def read_question(first="\n> ", rest="  "):
    """One question, which may run to several lines. None means stop.

    input() hands back a single line, so a pasted block used to arrive as its
    first line plus a queue of leftovers asked as further questions - and the
    first empty line inside the paste ended the session outright. Here an empty
    line ENDS THE QUESTION and leaving is explicit instead. That costs a second
    Enter on a one-line question, and makes pasting a question work at all.

    A block with a blank line INSIDE it still splits there; nothing in-band can
    tell that apart from the end of a question. Send those with --ask and a
    quoted heredoc, which has no terminator to collide with.
    """
    lines = []
    while True:
        try:
            line = input(first if not lines else rest)
        except (EOFError, KeyboardInterrupt):
            return None
        # A terminal in bracketed-paste mode wraps a paste in these; a readline
        # that does not consume them leaves them in the text, where they become
        # part of the question and the rest of the first line goes missing.
        line = line.replace("\x1b[200~", "").replace("\x1b[201~", "")
        if not lines and line.strip().lower() in ("quit", "exit"):
            return None
        if not line.strip():
            if not lines:
                continue            # a stray Enter at an empty prompt
            return "\n".join(lines).strip()
        lines.append(line)


def chat(model, tokenizer, device, max_new_tokens, temperature, system,
         think=False):
    """A plain read-generate loop. Each turn stands alone - no history is kept.

    Deliberately stateless: this exists to check what the model does with a
    prompt, and accumulated history would change the answer between two
    identical questions.
    """
    log("")
    log("Type or paste a question. An empty line sends it.")
    log("`quit`, Ctrl-D or Ctrl-C to stop.")
    log("-" * 70)
    while True:
        question = read_question()
        if question is None:
            log("\nbye")
            return
        log(f"[{len(question.splitlines())} lines, {len(question)} chars received]")
        try:
            generate(model, tokenizer, question, device,
                     max_new_tokens=max_new_tokens,
                     temperature=temperature, system=system, stream=True,
                     think=think)
        except KeyboardInterrupt:
            log("\n[stopped]")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base model, and talk to the result.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter", metavar="DIR",
                        help="LoRA adapter to merge. A local directory, or an "
                             "s3:// URI, which is fetched into ~/.cache/kd/s3 "
                             "and reused next time.")
    source.add_argument("--merged", metavar="DIR",
                        help="An already-merged model; skip merging and just "
                             "load it. Also accepts an s3:// URI.")

    parser.add_argument("--base", metavar="ID",
                        help="Base model. Default: whatever the adapter records.")
    parser.add_argument("--out", metavar="DIR",
                        help="Write the merged model here. Without this the merge "
                             "happens in memory only.")
    parser.add_argument("--vocab-size", type=int, default=None, metavar="N",
                        help="Trim the base to N rows before merging. Only needed "
                             "when it cannot be worked out - see the notes below.")

    parser.add_argument("--ask", metavar="TEXT",
                        help="Ask one question and exit. '-' reads it from stdin.")
    parser.add_argument("--chat", action="store_true",
                        help="Interactive prompt loop")
    parser.add_argument("--system", metavar="TEXT", default=None,
                        help="System turn to prepend. Use the same one training "
                             "used, or none.")

    parser.add_argument("--think", action="store_true",
                        help="Show the reasoning trace on models that have one "
                             "(Qwen3 and relatives). Off by default, and a no-op "
                             "on models without a thinking mode.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 is greedy and reproducible (default)")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--dtype", default="auto", choices=sorted(DTYPES))
    args = parser.parse_args(argv)

    if args.merged and args.out:
        parser.error("--merged is already merged; --out has nothing to write")
    if not (args.out or args.ask or args.chat):
        parser.error("nothing to do: pass --out to save, or --ask / --chat to "
                     "talk to it")

    # s3:// anywhere a path is accepted, fetched before anything looks at disk.
    if args.merged:
        merged = localise(args.merged, "merged model")
        if not os.path.isdir(merged):
            raise SystemExit(f"xx  no such merged model directory: {merged}")
        model, tokenizer, device = load_merged(merged, args.device, args.dtype)
    else:
        from kd import paths
        adapter = localise(paths.adapter_dir_of(args.adapter), "adapter")
        if not os.path.isdir(adapter):
            raise SystemExit(f"xx  no such adapter directory: {adapter}")
        model, tokenizer, device = merge(
            adapter, base_id=localise(args.base, "base") if args.base else None,
            vocab=args.vocab_size, device=args.device, dtype=args.dtype)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        log(f"==> writing the merged model to {args.out}")
        model.save_pretrained(args.out)
        # The tokenizer goes with it, always. A merged model without one is a
        # directory nobody can load without knowing which tokenizer it wants.
        tokenizer.save_pretrained(args.out)
        size = sum(f.stat().st_size for f in os.scandir(args.out) if f.is_file())
        log(f"    {size / 1e9:.2f} GB")
        log(f"    load it with:  AutoModelForCausalLM.from_pretrained('{args.out}')")

    if args.ask:
        question = sys.stdin.read() if args.ask == "-" else args.ask
        print(generate(model, tokenizer, question, device,
                       max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, system=args.system,
                       think=args.think))
    if args.chat:
        chat(model, tokenizer, device, args.max_new_tokens, args.temperature,
             args.system, think=args.think)
    return 0


if __name__ == "__main__":
    sys.exit(main())
