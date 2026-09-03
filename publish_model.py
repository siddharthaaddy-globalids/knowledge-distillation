"""Publish a distilled student to the Hugging Face Hub, as an adapter and a merged model.

Produces two repos from one training run:

    <repo>-lora    the LoRA adapter alone (small; needs the base model at load time)
    <repo>         the merged model, ready to load with AutoModelForCausalLM

The merge is performed by `transformers` itself, so the merged checkpoint carries
the canonical key layout for its architecture. That matters: a merged file written
by another framework keeps that framework's naming, and `from_pretrained` will
silently randomly-initialise every key it cannot map rather than raising.

    python publish_model.py --repo my-org/qwen3.5-0.8b-finance
    python publish_model.py --repo my-org/qwen3.5-0.8b-finance --dry-run
    python publish_model.py --repo my-org/... --adapter-only
    python publish_model.py --repo my-org/... --private

Authentication comes from `hf auth login` or the HF_TOKEN environment variable.

The merged model is verified locally before anything is uploaded: it is loaded
back with plain `AutoModelForCausalLM` and asked to generate. Publishing is
refused if the result is not coherent language.
"""

import argparse
import json
import pathlib
import shutil
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_CARD = """---
base_model: {base}
library_name: peft
tags:
- lora
- peft
- knowledge-distillation
- gkd
---

# {name} (LoRA adapter)

LoRA adapter distilled from **{teacher}** into **{base}** using
[Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649) (GKD).

This repo holds the **adapter only**. For a single ready-to-run checkpoint see
[`{merged_repo}`]({hub}/{merged_repo}).

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{base}", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "{repo}")
tok = AutoTokenizer.from_pretrained("{repo}")

messages = [{{"role": "user", "content": "How does compound interest work?"}}]
inputs = tok.apply_chat_template(messages, return_tensors="pt",
                                 add_generation_prompt=True)
print(tok.decode(model.generate(inputs, max_new_tokens=128)[0]))
```

## Training

| | |
|---|---|
| Student (base) | `{base}` |
| Teacher | `{teacher}` |
| Dataset | `{dataset}` |
| Method | GKD (on-policy, JSD loss) |
| LoRA rank / alpha | {r} / {alpha} |
| Target modules | {targets} |
{extra_rows}
"""

MERGED_CARD = """---
base_model: {base}
library_name: transformers
pipeline_tag: text-generation
tags:
- knowledge-distillation
- gkd
---

# {name}

**{base}** distilled from **{teacher}** with
[Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649) (GKD),
with the LoRA adapter merged in. Loads directly with `AutoModelForCausalLM` -
no PEFT required.

The LoRA adapter alone is at [`{adapter_repo}`]({hub}/{adapter_repo}).

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo}", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("{repo}")

messages = [{{"role": "user", "content": "How does compound interest work?"}}]
inputs = tok.apply_chat_template(messages, return_tensors="pt",
                                 add_generation_prompt=True)
print(tok.decode(model.generate(inputs, max_new_tokens=128)[0]))
```

## Training

| | |
|---|---|
| Student (base) | `{base}` |
| Teacher | `{teacher}` |
| Dataset | `{dataset}` |
| Method | GKD (on-policy, JSD loss) |
| LoRA rank / alpha | {r} / {alpha} |
| Target modules | {targets} |
{extra_rows}

## Notes

Merged with `transformers`, so the checkpoint uses this architecture's canonical
tensor names and loads without key remapping.
"""

PROBES = [
    "How does compound interest work? Explain briefly.",
    "What is the difference between a Roth IRA and a traditional IRA?",
]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Publish a distilled student to the Hub, as adapter and merged model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("-r", "--repo", required=True,
                    help="Target repo for the MERGED model, e.g. my-org/qwen3.5-0.8b-finance. "
                         "The adapter goes to the same name with a -lora suffix.")
    ap.add_argument("--adapter-repo", default=None,
                    help="Override the adapter repo name (default: <repo>-lora)")
    ap.add_argument("-a", "--adapter", default=None,
                    help="Adapter directory to publish (default: read from --config)")
    ap.add_argument("-c", "--config", default="configs/finance.yaml",
                    help="Training config, for provenance metadata and the adapter "
                         "path (default: configs/finance.yaml)")
    ap.add_argument("--adapter-only", action="store_true",
                    help="Publish only the adapter; skip the merged model")
    ap.add_argument("--merged-only", action="store_true",
                    help="Publish only the merged model; skip the adapter")
    ap.add_argument("--private", action="store_true", help="Create the repos as private")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"],
                    help="Dtype for the merged checkpoint (default: bfloat16)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and verify locally, print the cards, upload nothing")
    ap.add_argument("--keep", default=None, metavar="DIR",
                    help="Keep the merged model in DIR instead of a temp directory")
    return ap.parse_args()


def load_provenance(config_path, adapter_dir):
    """Gather what the model cards should say, from the adapter and training config."""
    info = {"teacher": "unknown", "dataset": "unknown", "extra": []}
    cfg_file = pathlib.Path(config_path)
    if cfg_file.exists():
        try:
            import kd_config
            cfg = kd_config.load_config(str(cfg_file))
            info["teacher"] = cfg["models"]["teacher"]
            if cfg["models"].get("teacher_adapter"):
                info["teacher"] += f" + {cfg['models']['teacher_adapter']}"
            info["dataset"] = cfg["dataset"]["source"]
            t, g = cfg.get("training", {}), cfg.get("gkd", {})
            info["extra"] = [
                ("Steps", t.get("max_steps")),
                ("Effective batch", (t.get("batch_size", 1)
                                     * t.get("gradient_accumulation_steps", 1))),
                ("Learning rate", t.get("learning_rate")),
                ("GKD lmbda / beta", f"{g.get('lmbda')} / {g.get('beta')}"),
            ]
        except Exception as exc:
            print(f" !! could not read {config_path}: {exc}")

    acfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    info["base"] = acfg.get("base_model_name_or_path", "unknown")
    info["r"] = acfg.get("r", "?")
    info["alpha"] = acfg.get("lora_alpha", "?")
    targets = acfg.get("target_modules") or []
    info["targets"] = ", ".join(f"`{t}`" for t in sorted(targets)) if targets else "n/a"
    return info


def verify(model_dir, tokenizer):
    """Load the merged model with plain transformers and confirm it emits language."""
    import math
    model, load_info = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, low_cpu_mem_usage=True, output_loading_info=True
    )
    model.eval()
    problems = []

    missing = [k for k in (load_info.get("missing_keys") or [])
               if not k.endswith(("rotary_emb.inv_freq", "lm_head.weight"))]
    if missing:
        problems.append(f"{len(missing)} weight(s) did not load: {missing[:4]}")

    for probe in PROBES:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": probe}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**ids).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
        uniform = math.log(logits.numel())
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=40, do_sample=False,
                                 repetition_penalty=1.1,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        answer = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n  Q: {probe}")
        print(f"     entropy {entropy:.2f} / {uniform:.2f} uniform")
        print(f"  A: {answer.strip()[:220]}")
        if entropy > 0.80 * uniform:
            problems.append(f"near-uniform output on {probe!r} - not predicting language")

    del model
    return problems


def main():
    args = parse_args()
    if args.adapter_only and args.merged_only:
        raise SystemExit("--adapter-only and --merged-only are mutually exclusive")

    adapter_dir = args.adapter
    if adapter_dir is None:
        import kd_config
        cfg = kd_config.load_config(args.config)
        adapter_dir = str(pathlib.Path(cfg["project"]["output_dir"]) / "final_adapter")
    adapter_dir = pathlib.Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"No adapter_config.json in {adapter_dir}")

    merged_repo = args.repo
    adapter_repo = args.adapter_repo or f"{merged_repo}-lora"
    hub = "https://huggingface.co"

    bar = "=" * 78
    print(bar)
    print("  Publish distilled model")
    print(bar)
    print(f"   adapter dir  : {adapter_dir}")
    print(f"   adapter repo : {adapter_repo}")
    print(f"   merged repo  : {merged_repo}")

    info = load_provenance(args.config, adapter_dir)
    print(f"   base model   : {info['base']}")
    print(f"   teacher      : {info['teacher']}")
    print(f"   dataset      : {info['dataset']}")

    extra_rows = "".join(f"| {k} | {v} |\n" for k, v in info["extra"] if v is not None)
    fields = dict(base=info["base"], teacher=info["teacher"], dataset=info["dataset"],
                  r=info["r"], alpha=info["alpha"], targets=info["targets"],
                  extra_rows=extra_rows.rstrip(), hub=hub,
                  adapter_repo=adapter_repo, merged_repo=merged_repo)

    api = None
    if not args.dry_run:
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            who = api.whoami()
            print(f"   authenticated: {who.get('name')}")
        except Exception:
            raise SystemExit(
                "Not authenticated with the Hub. Run `hf auth login`, or set HF_TOKEN."
            )

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir) if (adapter_dir / "tokenizer.json").exists() else info["base"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 1. adapter ------------------------------------------------------- #
    if not args.merged_only:
        print(f"\n[1/3] Adapter -> {adapter_repo}")
        card = ADAPTER_CARD.format(name=adapter_repo.split("/")[-1],
                                   repo=adapter_repo, **fields)
        (adapter_dir / "README.md").write_text(card, encoding="utf-8")
        if args.dry_run:
            print("  (dry run) README.md written; not uploading")
        else:
            api.create_repo(adapter_repo, private=args.private, exist_ok=True)
            api.upload_folder(folder_path=str(adapter_dir), repo_id=adapter_repo,
                              commit_message="Publish distilled LoRA adapter")
            print(f"  -> {hub}/{adapter_repo}")

    if args.adapter_only:
        print("\nDone (adapter only).")
        return 0

    # ---- 2. merge --------------------------------------------------------- #
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    workdir = pathlib.Path(args.keep) if args.keep else pathlib.Path(tempfile.mkdtemp())
    merged_dir = workdir / "merged"
    print(f"\n[2/3] Merging adapter into {info['base']} ({args.dtype})...")
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(info["base"], dtype=dtype,
                                                low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    del model, base
    size = sum(f.stat().st_size for f in merged_dir.glob("*.safetensors"))
    print(f"  -> {merged_dir} ({size / 1e9:.2f} GB)")

    # ---- 3. verify then upload -------------------------------------------- #
    print("\n[3/3] Verifying the merged checkpoint before upload...")
    problems = verify(str(merged_dir), tokenizer)
    if problems:
        print("\n" + bar)
        print("  REFUSING TO PUBLISH - the merged model is not sound")
        print(bar)
        for p in problems:
            print(f"  * {p}")
        return 2
    print("\n  merged checkpoint verified: all weights loaded, output is language")

    card = MERGED_CARD.format(name=merged_repo.split("/")[-1], repo=merged_repo, **fields)
    (merged_dir / "README.md").write_text(card, encoding="utf-8")

    if args.dry_run:
        print("\n--- merged model card ---")
        print(card[:1200])
        print(f"\n(dry run: nothing uploaded; merged model at {merged_dir})")
        return 0

    api.create_repo(merged_repo, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(merged_dir), repo_id=merged_repo,
                      commit_message="Publish merged distilled model")
    print(f"  -> {hub}/{merged_repo}")

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n" + bar)
    print("  ALL OK - published")
    print(bar)
    print(f"   adapter : {hub}/{adapter_repo}")
    print(f"   merged  : {hub}/{merged_repo}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
