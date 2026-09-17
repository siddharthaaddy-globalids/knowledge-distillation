# Running the packed student

The pipeline produces three things you can run. This is about the third one —
the W4A16 checkpoint — and where it will and will not work.

| artifact | size | what it is | runs on |
| --- | --- | --- | --- |
| `final_adapter/` | 0.7 GB | LoRA deltas; needs Qwen3-8B underneath | anything that fits 8B at bf16 |
| `merged/` | 15 GB | the dense student, no PEFT | anything that fits 8B at bf16 |
| `quantized/` | 5.7 GB | the same student packed to 4 bits | **CUDA only** |

`quantized/` is a complete standalone model — `kd quantize` merges the adapter
into the base first and packs *that*, so there is no adapter to apply and no
base to fetch. What it is not is portable: the weights are int4 with per-group
scales and a `quantization_config` naming compressed-tensors, and unpacking them
needs kernels that exist on CUDA and nowhere else.

For the adapter-and-merge route, and everything `infer.sh` can do, see
[INFERENCE.md](INFERENCE.md).

---

## A pod to try it on

Serving the student needs a fraction of what scoring it did — the 14B teacher
was the reason for a 48 GB card, and the teacher is not here.

| | |
| --- | --- |
| GPU | **16 GB is enough**; 20–24 GB is comfortable |
| Volume | 30 GB — 5.7 for the model, the rest for vLLM and torch |
| Template | any stock PyTorch image |
| HTTP Ports | add **8000** at creation time, only if you want to reach it from outside |

Roughly $0.20–0.45/hour, against $2.79 for the card that trained it.

Measured on a 19.55 GiB card, from vLLM's own startup accounting:

```
weights + non-torch   5.98 GiB
peak activation       1.74 GiB
CUDA graphs           0.27 GiB
KV cache              8.90 GiB
```

That KV cache is about seven concurrent 8k-token conversations. On a 12 GB card
it still runs with `max_model_len=4096`; on 8 GB it does not — weights plus
overhead fill the device before any cache is allocated.

---

## Setup

No repository clone. Nothing here imports `kd`, which also means none of the
dependency grief in [POD-FAILURES.md](POD-FAILURES.md) applies: vLLM reads
compressed-tensors natively, so there is no llm-compressor, no transformers pin
to satisfy, and no torchvision to repair.

```bash
pip install --break-system-packages vllm awscli

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...

aws s3 cp --recursive \
  s3://enlibra/dss/dev/runs/20260818_215840_neuroscience_8f33eb7cf609/outputs/gkd/runs/<run-id>/quantized \
  /workspace/enlibra-8b-w4a16

ls -la /workspace/enlibra-8b-w4a16/
```

That listing must show `config.json`, `model.safetensors`, `tokenizer.json`,
`tokenizer_config.json`, `chat_template.jinja` and `kd-quant.json`. A missing
`config.json` means a partial copy — rerun the same command, it resumes.

**Check the destination spelling.** A typo in the directory name does not fail
here; it fails later, inside vLLM, as `Repo id must be in the form
'repo_name' or 'namespace/repo_name'`. That error means the local path did not
exist, so transformers fell back to treating it as a Hub id. It never means what
it says.

---

## A terminal loop

Save as `chat.py`:

```python
"""Ask the packed student a few questions, one at a time, in a terminal.

    python3 chat.py [model-dir]

Loads once - a minute or two the first time - then stays interactive. Blank
line, Ctrl+D or Ctrl+C to quit.

Greedy by default, which is how the arena scored this model: sampling would
give a different answer each run and none of them comparable to the report.
"""

import sys

from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/workspace/enlibra-8b-w4a16"

llm = LLM(MODEL, max_model_len=8192, gpu_memory_utilization=0.85)
sampling = SamplingParams(max_tokens=512, temperature=0.0)

print(f"\nready: {MODEL}")
print("blank line, Ctrl+D or Ctrl+C to quit.\n")

while True:
    try:
        prompt = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not prompt:
        break

    out = llm.chat([[{"role": "user", "content": prompt}]], sampling)
    print("\n" + out[0].outputs[0].text.strip() + "\n")
```

```bash
python3 chat.py                              # the default path
python3 chat.py /workspace/some-other-model  # or any checkpoint directory
```

**Write it to a file rather than piping it in.** `python3 - <<'PY'` feeds the
*script* through stdin, so `input()` reaches end-of-file immediately and the
loop exits without ever asking anything. If you build the file with a heredoc,
the closing `PY` must be at column 0 — indent it and the shell keeps reading,
and every prompt you type becomes part of the script.

**A blank line quits.** Do not press Enter to test whether it is ready.

### Startup, and what "ready" looks like

Two minutes on a cold pod, and most of it prints nothing useful:

```
Loading safetensors checkpoint shards: 100% ...      seconds
Dynamo bytecode transform time: 11.31 s
Compiling a graph for compile range (1, 8192) takes 27.06 s
Capturing CUDA graphs (PIECEWISE): 100% ...
Capturing CUDA graphs (FULL): 100% ...
```

None of those mean it is ready. The signal is the script's own line:

```
ready: /workspace/enlibra-8b-w4a16

you>
```

The `torch.compile` cache lands in `/root/.cache/vllm`, on the container disk,
so it dies with the pod. Keep it on the volume and the second launch takes about
twenty seconds instead of two minutes:

```bash
export VLLM_CACHE_ROOT=/workspace/vllm-cache
```

---

## As a server instead

```bash
tmux new -s serve
vllm serve /workspace/enlibra-8b-w4a16 \
    --served-model-name enlibra-8b \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

From inside the pod:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "enlibra-8b",
  "messages": [{"role": "user", "content": "Explain in two sentences what myelin does."}],
  "max_tokens": 512
}'
```

From anywhere, if port 8000 was exposed when the pod was created — RunPod
proxies it at `https://<pod-id>-8000.proxy.runpod.net`. It is an
OpenAI-compatible endpoint, so any client works: point `base_url` at that URL
and send any string as the API key.

`--max-model-len 8192` matches what the arena scored at. Leave it off and vLLM
sizes the context to the model's full 32k, which buys a much smaller KV cache
for a length nothing here uses.

---

## First prompts

Two are worth running before anything else, because the training run watched
them every 100 steps — they are `benchmark_prompts` in the profile:

```
Who are you?
Explain in two sentences what myelin does.
```

The first tests the persona, which is 15 rows against 1369 neuroscience ones —
1.1% of the mix, and the profile is candid that this may be too thin to survive
two epochs. Answers that name **enLibra**, decline to name a base model, and
turn down non-neuroscience work are the ones the identity rows were teaching.
"I am Qwen, created by Alibaba Cloud" means they did not land.

Then try prompts that are *not* in `data/enlibra-neuroscience/identity.jsonl` —
"Introduce yourself", "Which company made you?", "Can you help me debug some
Python?". Passing the trained wording and failing the paraphrase is memorisation
rather than a persona, and the fix is more identity rows phrased differently,
not repeats: deduplication drops those.

For the curriculum itself, expect a ~300-token explanation ending in an
`<Answer>` tag — that is the shape it was trained to produce, and why
`arena_max_new_tokens` is 8192 rather than something tight.

---

## On a Mac

The packed checkpoint will not run there. It is not a matter of size — 5.7 GB
would suit a 16 GB Mac mini perfectly — but of format: there is no MPS kernel
for int4 compressed-tensors, and llm-compressor has no macOS build at all. The
download succeeds and the load fails.

Two ways to actually get this model onto a Mac:

**The dense student, through `infer.sh`.** Merges base and adapter in memory,
or writes the merge once:

```bash
./infer.sh --adapter s3://<bucket>/<prefix>/runs/<run-id>/final_adapter \
    --out ~/models/enlibra-8b --device mps
./infer.sh --merged ~/models/enlibra-8b --chat --device mps
```

16.4 GB of bf16 weights, so ~18–19 GB of unified memory: a 32 GB machine, not a
16 GB one.

**A Mac-native 4-bit.** Convert the dense merge to MLX or GGUF — and do the
conversion on the pod, where the merge already exists and the memory is free,
then ship the ~5 GB result:

```bash
pip install mlx-lm
mlx_lm.convert --hf-path /workspace/runs/<run-id>/merged -q --q-bits 4 \
               --mlx-path ./enlibra-8b-mlx-4bit
```

If `merged/` is gone — `quantization.keep_merged` is false, so the pipeline
deletes it once the packing is done — rebuild it from the adapter in a couple of
CPU-minutes with the `infer.sh --out` command above.

**Those numbers will not match the report.** An MLX or GGUF build is quantized
by a different algorithm with different rounding, so it is a different model
from the `distilled-w4a16` player the arena scored. When the measured artifact
is what matters, that is the CUDA one.

---

## See also

- [INFERENCE.md](INFERENCE.md) — `infer.sh`, merging, and the options it passes through
- [POD-FAILURES.md](POD-FAILURES.md) — every error seen on these pods and what it meant
- [RUNBOOK-14B.md](RUNBOOK-14B.md) — producing these artifacts in the first place
- [EXPLANATION-SCORING.md](EXPLANATION-SCORING.md) — how to read what the arena said about them
