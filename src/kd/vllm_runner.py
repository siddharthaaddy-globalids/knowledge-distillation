"""
The arena's generation, done by vLLM instead of one `model.generate()` at a time.

    python -m kd arena --config ... --engine vllm

WHY THIS EXISTS
---------------
`kd.arena._answer_all` walks the held-out set one question at a time, and the
enlibra profiles allow 8192 new tokens per answer. Four players over ~140
questions is 560 sequential generations, and it is far and away the longest
thing the pipeline does - longer than the training it is scoring.

Every one of those prompts is known before the first token is generated, which
is the exact shape vLLM's continuous batching is for. Same questions, same
greedy decode, the whole set in flight at once.

WHY A SUBPROCESS
----------------
`kd.arena.play` loads its players ONE AT A TIME and frees each before the next,
because an 8B student and a 14B teacher do not fit on one card together. That
contract is the reason the arena runs on a single rented GPU at all.

vLLM makes it hard to keep. It preallocates a KV cache to
`gpu_memory_utilization` of the card and holds it in worker processes and a
NCCL group; tearing all of that down inside a live interpreter needs a
different incantation in every release, and the failure mode is not an
exception but the NEXT player meeting a card that is still 85% full.

A subprocess has none of that problem. The operating system frees the memory,
completely, every time, in every version. It costs one model load per player -
which the HF path pays anyway, because it also loads each player exactly once.

WHAT IS AND IS NOT SHARED WITH THE HF PATH
------------------------------------------
Prompts are rendered and TOKENISED in the parent, by the tokenizer the arena
already chose (the adapter's own, not the Hub's), and the token ids are what
crosses into the subprocess. Not the text. That removes the whole class of bug
where two engines disagree about a chat template or about whether to add a BOS,
and makes the two paths comparable by construction rather than by inspection.

Stop tokens are deliberately left to the model's own generation_config.json,
which vLLM reads. Qwen3 ships `eos_token_id: [151645, 151643]` there and names
only 151643 in config.json, so overriding it - which `generation_config="vllm"`
would do - loses `<|im_end|>` and every answer runs to the 8192-token ceiling
instead of stopping. That is a slow, expensive, entirely silent failure, and
it is why this module does not touch that setting.
"""

import json
import os
import subprocess
import sys

# Sized to the job rather than to the model: the KV cache vLLM preallocates is
# proportional to max_model_len, and a Qwen3 default of 40960 reserves memory
# for a context this run will never use. Rounded up so a one-token difference
# between runs does not change the allocation.
LENGTH_GRANULARITY = 256

# LLM() keywords passed through from the config. Every one of these has been
# stable across vLLM releases for a long time; anything newer is deliberately
# not exposed, because a keyword this pipeline cannot rely on is one that turns
# an arena into a TypeError on somebody else's pod.
ENGINE_OPTIONS = ("gpu_memory_utilization", "max_model_len", "enforce_eager",
                  "tensor_parallel_size", "trust_remote_code", "swap_space")


def available():
    """True when vLLM can be imported in this interpreter."""
    import importlib.util

    return importlib.util.find_spec("vllm") is not None


def unavailable_reason():
    """Why `--engine vllm` cannot run here, in a sentence, or None if it can."""
    if available():
        return None
    if sys.platform in ("win32", "darwin"):
        where = "Windows" if sys.platform == "win32" else "macOS"
        return (f"vLLM publishes no {where} wheel, so engine `vllm` cannot run "
                f"here.\n"
                f"      Ask for the other engine explicitly - it is a visible "
                f"choice, not a silent fallback:\n"
                f"          ./run.sh --config <profile> --set "
                f"evaluation.engine=hf ...\n"
                f"          kd arena --engine hf ...\n"
                f"      The two engines do not agree token for token, so a "
                f"local score and a pod score\n"
                f"      are not comparable - which is the reason you are asked "
                f"rather than defaulted.")
    return ("vLLM is not installed. It is an optional extra because it pulls a "
            "CUDA build of torch that the default CPU pin would fight:\n"
            "      uv pip install --index-url https://download.pytorch.org/whl/cu128 "
            "--extra-index-url https://pypi.org/simple vllm")


def plan_length(prompts, max_new_tokens, requested=None):
    """The context window to give the engine: the longest prompt plus the ceiling.

    `requested` wins when the config names one. Otherwise this is measured from
    the questions actually being asked, which is both smaller than the model's
    declared maximum and guaranteed to fit every one of them.
    """
    if requested:
        return int(requested)
    longest = max((len(ids) for ids in prompts), default=0)
    needed = longest + int(max_new_tokens)
    return -(-needed // LENGTH_GRANULARITY) * LENGTH_GRANULARITY


# --------------------------------------------------------------------------- #
# Parent side
# --------------------------------------------------------------------------- #
def complete(model, prompts, max_new_tokens, tokenizer=None, dtype=None,
             options=None, workdir=None, log=None):
    """Every prompt answered greedily, in one batch, in a subprocess.

    `prompts` is a list of token-id lists - see the header for why it is not a
    list of strings. Returns the completions as decoded text, in the order the
    prompts were given, which is what `kd.arena` scores.
    """
    import tempfile

    options = dict(options or {})
    workdir = workdir or tempfile.mkdtemp(prefix="kd-vllm-")
    os.makedirs(workdir, exist_ok=True)
    job_path = os.path.join(workdir, "job.json")
    out_path = os.path.join(workdir, "completions.json")

    options["max_model_len"] = plan_length(
        prompts, max_new_tokens, options.get("max_model_len"))
    job = {
        "model": str(model),
        "tokenizer": str(tokenizer) if tokenizer else None,
        "dtype": dtype,
        "max_new_tokens": int(max_new_tokens),
        "prompts": [list(map(int, ids)) for ids in prompts],
        "options": {k: v for k, v in options.items()
                    if k in ENGINE_OPTIONS and v is not None},
    }
    with open(job_path, "w", encoding="utf-8") as handle:
        json.dump(job, handle)
    if os.path.isfile(out_path):
        os.remove(out_path)     # so a crashed child cannot return stale answers

    if log:
        log.info(f"      vllm: {len(prompts)} prompts, up to {max_new_tokens} new "
                 f"tokens, context {options['max_model_len']}")

    # The package has to be importable in the child. An installed checkout is
    # the ordinary case; a bare source tree is not, and inheriting the path we
    # were imported from covers both without asking the caller which it is.
    env = dict(os.environ)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [package_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    # stdout and stderr are NOT captured. vLLM's load and its generation
    # progress bar go straight to the terminal, which on a forty-minute batch
    # is the difference between a running job and an apparently hung one.
    finished = subprocess.run(
        [sys.executable, "-m", "kd.vllm_runner", job_path, out_path], env=env)

    result = {}
    if os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as handle:
            result = json.load(handle)
    if result.get("error"):
        raise RuntimeError(f"the vllm worker failed: {result['error']}")
    if finished.returncode != 0:
        raise RuntimeError(
            f"the vllm worker exited {finished.returncode} without writing a "
            f"result. Its output is above; a killed worker is usually the card "
            f"running out of memory, which evaluation.vllm.gpu_memory_utilization "
            f"lowers.")

    completions = result.get("completions") or []
    if len(completions) != len(prompts):
        raise RuntimeError(
            f"the vllm worker answered {len(completions)} of {len(prompts)} "
            f"prompts. Scoring a short set as if it were the whole one would "
            f"report a number nobody could reproduce.")
    return completions


# --------------------------------------------------------------------------- #
# Child side
# --------------------------------------------------------------------------- #
def run_job(job):
    """Load the engine, answer every prompt, return the completions as text."""
    from vllm import LLM, SamplingParams

    engine = dict(job.get("options") or {})
    if job.get("tokenizer"):
        # Named explicitly rather than left to the model directory, so the
        # detokenisation matches the tokenizer the parent encoded with even when
        # the player is a stock hub checkpoint.
        engine["tokenizer"] = job["tokenizer"]
    if job.get("dtype"):
        engine["dtype"] = job["dtype"]

    llm = LLM(model=job["model"], **engine)
    sampling = SamplingParams(
        n=1,
        # Greedy, always, for the same reason kd.arena._generate is greedy: an
        # accuracy measured with sampling is a different number every run, and
        # the difference between two runs of one model reads as a difference
        # between two models.
        temperature=0.0,
        max_tokens=int(job["max_new_tokens"]),
        skip_special_tokens=True,
        detokenize=True,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in job["prompts"]], sampling)
    # vLLM returns completions in request order, but says so as a property of
    # the API rather than of any one release, and the arena lines answers up
    # with questions by index. Sorting by the request id it assigns makes that
    # alignment something this file guarantees rather than something it assumes.
    ordered = sorted(outputs, key=lambda out: int(out.request_id))
    return [out.outputs[0].text.strip() for out in ordered]


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    if len(argv) != 2:
        print("usage: python -m kd.vllm_runner <job.json> <out.json>",
              file=sys.stderr)
        return 2
    job_path, out_path = argv

    with open(job_path, encoding="utf-8") as handle:
        job = json.load(handle)

    try:
        payload = {"completions": run_job(job)}
    except BaseException as exc:  # noqa: BLE001 - the parent must see every cause
        # Written to the result file rather than only raised, because the parent
        # reads a file and not a traceback. Without this an out-of-memory kill
        # and a typo in a model id are the same empty non-zero exit.
        import traceback

        traceback.print_exc()
        payload = {"error": f"{type(exc).__name__}: {exc}"}
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return 1

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
