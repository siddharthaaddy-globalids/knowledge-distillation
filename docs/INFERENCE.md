# Using a trained model

A run produces a **LoRA adapter**, not a model. `./infer.sh` merges it into the
base and lets you talk to the result.

```bash
./infer.sh --adapter ./downloaded-adapter --base Qwen/Qwen2.5-1.5B-Instruct --ask "Who are you?"
```

That is the fully explicit form: which adapter, which base model, what to ask.
Everything below is a way of typing less of it.

---

## Typing less

**The base is usually already known.** PEFT writes it into the adapter's own
`adapter_config.json` when it saves, so `--base` is only needed when that record
is wrong or the model has moved:

```bash
./infer.sh --adapter ./downloaded-adapter --ask "Who are you?"
```

**The adapter is usually the one you just trained.** With none named it takes the
newest under `runs/`, and prints which:

```bash
./infer.sh --ask "Who are you?"
```

It always says what it resolved, so you never have to guess:

```
   newest adapter: runs/20260910T091455Z-.../final_adapter
==> base    : Qwen/Qwen2.5-1.5B-Instruct
==> adapter : runs/20260910T091455Z-.../final_adapter
```

---

## Straight from S3

`--adapter` takes an `s3://` URI as readily as a path:

```bash
./infer.sh --adapter s3://enlibra/dss/dev/runs/<run>/outputs/gkd/runs/<run-id>/final_adapter --chat
```

It fetches into `~/.cache/kd/s3`, keyed by bucket and key, and reuses it next
time — the same cache the pipeline uses, so an adapter a run already pulled down
is not pulled twice. Credentials come from the environment, as everywhere else.

`--merged` and `--base` accept `s3://` too.

---

## The three things it does

```bash
./infer.sh --ask "What are stars formed from?"   # ask one question
./infer.sh --chat                                # keep asking
./infer.sh --out ./merged                        # write a standalone model
```

They combine — `--out ./merged --ask "..."` saves *and* answers.

**Asking** is greedy by default, so the same question gives the same answer every
time. `--temperature 0.7` samples instead. `--ask -` reads from stdin, for
anything long.

**Chat** is a prompt loop; empty line or Ctrl-C leaves. Each turn stands alone —
no history is kept, deliberately, because accumulated history would change the
answer between two identical questions.

**Saving** writes an ordinary checkpoint that loads with no PEFT involved:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./merged")
tok = AutoTokenizer.from_pretrained("./merged")
```

The tokenizer goes with it, so the directory is self-contained. Talk to it later
without merging again:

```bash
./infer.sh --merged ./merged --chat
```

---

## Adapter or merged model?

| | Adapter | Merged |
|---|---|---|
| Size | tens of MB | the whole model |
| Loading | needs `peft` + the base | plain `from_pretrained` |
| Swappable | yes, several on one base | no |
| Good for | keeping, comparing, iterating | serving, sharing, deploying |

Merging adds the LoRA's low-rank deltas into the base weights once, instead of
`peft` doing it in memory on every load. Nothing is lost — the merged model
computes exactly what the adapter did.

Keep the adapter. Merge when something needs a model that does not know what a
LoRA is.

---

## Options worth knowing

| | |
|---|---|
| `--base ID` | Override the base the adapter names. |
| `--system "..."` | Prepend a system turn. Use the same one training used, or none. |
| `--temperature 0.7` | Sample instead of greedy. Default `0` is reproducible. |
| `--max-new-tokens 2048` | Default 512. Raise it if answers are cut off mid-sentence. |
| `--device cuda\|mps\|cpu` | Default picks the best available. |
| `--dtype bfloat16` | Default is bfloat16 on CUDA, float32 elsewhere. |
| `--vocab-size N` | Only if it cannot work the width out — see below. |

`./infer.sh --help` lists them all.

---

## If it complains about a size mismatch

```
size mismatch for base_model.model.lm_head.weight:
  copying a param with shape torch.Size([151665, 1536]) ...
  the shape in current model is torch.Size([151936, 1536])
```

Stock Qwen checkpoints pad `vocab_size` up to a multiple of 128 for tensor
alignment — 151936 against a tokenizer with 151665 real tokens. A teacher that
was fine-tuned through `resize_token_embeddings()` has had that padding trimmed,
and `kd.train` trims the student to match before training. So the adapter is
built against the narrower width.

`infer.sh` works this out on its own from the adapter, or from the run bundle
beside it. It only needs telling when the adapter has been moved away from its
bundle:

```bash
./infer.sh --adapter ./downloaded-adapter --vocab-size 151665 --chat
```

The number is the teacher's `vocab_size`, in the `config.json` of the checkpoint
the run distilled from.

---

## Reading what comes back

The first thing worth checking after a real run is whether the identity took:

```bash
./infer.sh --ask "Who are you?"
```

The curriculum trains 15 persona rows against roughly a thousand questions —
about 1.4% of the mix. If the answer still begins *"I am Qwen, developed by
Alibaba Cloud"*, the persona did not survive, and the fix is more identity rows
in the export rather than anything in this repository.

For the questions the model was actually trained on, expect the tagged shape it
learned:

```
<Explanation>
...
</Explanation>
<Answer>:
C
</Answer>
```

An answer that trails off mid-sentence usually means `--max-new-tokens` is too
low, not that the model has nothing more to say.

## See also

- [START-HERE.md](START-HERE.md) — training, start to finish
- [CONFIG.md](CONFIG.md) — every setting
- `kd publish --repo <org>/<name>` — merge *and* push to the Hugging Face Hub
  with a model card, when that is where the merged model is headed
