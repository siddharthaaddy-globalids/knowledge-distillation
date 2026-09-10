# Start here

Clone the repository. Run one script. That is the whole thing.

```bash
git clone <GIT REPO URL> knowledge-distillation
cd knowledge-distillation
./run.sh
```

`run.sh` works out where it is running and does the right thing there:

| Where you run it | What it does |
|---|---|
| **Your Mac** (no GPU) | Installs everything, checks it, runs the **smoke test** — small stand-in models, the whole pipeline, minutes, free |
| **A RunPod pod** (GPU) | Installs everything, checks it, runs the **real training** |

There is a third thing it can do on request — `./run.sh full`, which trains on
all the data and runs the whole evaluation locally. See
[Part 1b](#part-1b--does-it-actually-learn-still-free-but-hours).

Same command in both places. The expensive one only happens on the machine that
is expensive anyway.

The one case it refuses to guess: if the machine **looks rented** (`RUNPOD_POD_ID`
or `KD_PRICE_PER_HOUR` is set) but no GPU is visible, it stops and asks. Running
a smoke test on a pod that is billing you would look like success while being the
worst outcome available — that is almost always the wrong pod template.

**Do the Mac first.** A RunPod GPU bills by the second from the moment it
starts, so everything that can be proven for free should be proven for free.

---

## Before you begin

Keep these three things in a note — you will paste them more than once.

```
GIT REPO URL     the https://... address of this repository
AWS ACCESS KEY   the pair that can read the teacher model from S3
RUNPOD API KEY   from https://console.runpod.io/user/settings
```

You do **not** need to install Python. `run.sh` installs `uv`, which handles
Python for you.

---

## Part 1 — On the Mac mini (free)

```bash
git clone <GIT REPO URL> knowledge-distillation
cd knowledge-distillation

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

./run.sh
```

That is it. The script installs uv, installs the project, reports what your
machine can do, checks whether it can actually reach the teacher in S3, prints
the plan for the real run, and then runs the full pipeline on small models.

Expect **20–40 minutes** the first time — most of it downloading about 5 GB of
stand-in models. Later runs are much faster.

### What you are looking for

Every stage saying `OK`, and at the end:

```
  run bundle : runs/2026...-enlibraQ3-8B-smoke-...
  adapter    : runs/2026...-enlibraQ3-8B-smoke-.../final_adapter
```

Along the way it prints the dataset it built, which should look like this:

```
  - curriculum-sft   937/937  (100.0% pass)
  - curriculum-rl    143/143  (100.0% pass)
  - identity          15/15   (100.0% pass)
  Total collected : 1095
```

**All three at 100%** is what matters. The smoke test builds the *whole* corpus,
not a sample of it, so this proves every row is readable and inside the token
budgets — on your machine, before a GPU is rented. Below 100% means rows are
being silently dropped and the real run would train on part of the data.

It still only *trains* on four samples. Rows built and rows trained on are
different numbers, and it is the first one that says whether the data is sound.

### The one thing to check carefully

Near the top of the output, under **remote inputs**:

```
 remote inputs (one listing each, nothing downloaded)
   models.teacher           OK    s3://enlibra/dss/dev/runs/.../models/rl/
```

If that says `FAIL`, **stop and fix it before renting a GPU.** The message tells
you which of the two problems it is. If it is `access denied`, new access keys
will not help — the permission is missing from your AWS user, not from your key.
Ask whoever administers the AWS account for `s3:ListBucket` on the bucket and
`s3:GetObject` on that prefix.

Everything else in Part 1 works without it. Only the real run needs it.

### Talk to what you trained

The smoke adapter is trained for two steps, so expect nonsense. The point is
that the plumbing works.

```bash
./run.sh ask "What are stars formed from?"
```

---

## Part 1b — Does it actually learn? (still free, but hours)

The smoke test proves nothing is broken. It does **not** tell you whether
distillation works on this data, because it trains on four samples.

If you want that answer before renting a GPU — and it is a good answer to have:

```bash
./run.sh full
```

This trains on **all 1087 rows for the full 300 steps**, then scores **all 137
held-out questions** with all three players. Everything is inherited from the
real profile — corpus, LoRA shape, schedule, token budgets — except the models,
which are one size down (Qwen3-1.7B → Qwen3-0.6B) because 19.0 GB does not fit
16 GB of memory.

Budget **several hours**. It is free, and safe to interrupt: checkpoints are
written every 25 steps, and each stage writes into the run bundle as it finishes.

**Watch the sample generations**, printed every 50 steps. They are the most
informative thing in the log — the student should go from noise to the
`<Explanation>…<Answer>` shape within the first hundred steps or so.

At the end you get the same table the pod will produce. Read it like this:

| | |
|---|---|
| **It answers** | Does the format transfer? Does the loss fall? Does distilled beat base on the answer key? |
| **It does not answer** | What accuracy the real 8B → 1.7B pair will reach. A 0.6B student copying a 1.7B teacher is a different problem. |

**Read the direction, not the number.** If distilled beats base here,
distillation works on this data. If it does not, it will not work on the pod
either — and you found that out for nothing.

### Can I force the real 8B teacher on the Mac?

No — and the arithmetic is worth seeing, because it is the same arithmetic that
decides which GPU you rent.

Both models are resident at once: the teacher frozen, the student training. So
the sum is what has to fit, *before* activations, optimizer state and the OS.

| Pairing | bf16 weights | On a 16 GB Mac |
|---|---|---|
| Qwen3-8B → Qwen3-1.7B | **19.0 GB** | does not fit — the teacher alone is 15.3 GB |
| Qwen3-4B → Qwen3-1.7B | 11.3 GB | runs, but swaps hard |
| Qwen3-1.7B → Qwen3-0.6B | 5.2 GB | comfortable — this is `./run.sh full` |

`kd.paths` treats 60% of RAM (9.6 GB here) as the working budget, because
activations and the KV cache sit on top of the weights. Past that macOS does not
refuse — it swaps, and the run gets mysteriously slow rather than failing
honestly.

**The useful middle ground** is a 4B teacher into the *real* 1.7B student. It
exercises the actual student, the actual LoRA target modules and the actual
per-step memory on the student side — none of which `./run.sh full` covers,
since that one shrinks both halves:

```bash
./run.sh train \
    --set models.teacher=Qwen/Qwen3-4B \
    --set s3.enabled=false \
    --set hardware.device=mps \
    --set training.max_steps=20 \
    --set limits.max_runtime_minutes=600
```

Expect it to be slow. It is a memory-shape test, not a training run.

**If you want to watch the 8B try anyway** — swap `Qwen3-4B` for `Qwen3-8B`
above. It downloads 15.3 GB and then thrashes; `preflight` warns first:

```
!! the teacher and student together are 19.0 GB of bfloat16 weights, against a
   9.6 GB budget (60% of this machine's 16 GB).
```

That warning is not a formality. Nothing you learn from pushing past it applies
to the pod, which has different memory and a different backend.

`--set s3.enabled=false` in both commands substitutes a stock Hub teacher for
the S3 one, so these work without any AWS access.

---

## Part 2 — On RunPod (paid)

### Start a pod

In the RunPod console:

1. **Deploy** → a GPU with **at least 48 GB** — an **L40S** or **RTX A6000**.
   A 24 GB card will not do; the two models are 19.0 GB of weights before
   anything else.
2. Template: any **PyTorch** template. This matters — the script keeps that
   template's CUDA build of torch instead of downloading its own, saving about
   five minutes of paid time.
3. Volume: **120 GB**, mounted at `/workspace`.
4. Note the **price per hour** shown.
5. Deploy, then **Connect → SSH**, and copy the command.

### Then, on the pod

```bash
cd /workspace
git clone <GIT REPO URL> knowledge-distillation
cd knowledge-distillation

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
export KD_PRICE_PER_HOUR=0.89        # the rate from step 4 above

tmux new -s kd
./run.sh
```

Same script. It sees the GPU and runs the real training.

**`tmux` is not optional.** If your connection drops, anything in the foreground
dies *and the pod keeps billing*. Inside tmux the run survives; reconnect by SSH
and type `tmux attach -t kd` to find it. The script will warn and ask before
continuing without it.

**`KD_PRICE_PER_HOUR`** is what makes the spending cap real. Without it, the run
has no idea what it is costing you. The script warns if it is missing.

Expect **1–3 hours**. You can close your laptop.

### When it finishes

```
  137 held-out questions, random baseline 25%

  player        accuracy      elo    +/-  unanswered
  teacher          79.6%     1081     20           0
  distilled        57.7%      993     22           0
  base             35.8%      926     23           4
```

`base` is the stock small model with no training — the control. `distilled` is
what you just made.

| | |
|---|---|
| distilled **above** base | Training worked. The gap to teacher is your headroom. |
| distilled **level with** base | The format transferred, the ability did not. |
| distilled **below** base | Training hurt. Something is wrong. |

A lower number than you hoped is not automatically a failure: the held-out
questions go up to 5 reasoning steps, where the training data stops at 3.

### Then terminate the pod

The script says this too, but it bears repeating: **nothing stops a pod you
started by hand.** It bills until you terminate it in the console — after the run
has finished, after you have closed the terminal. Go and check.

---

## The LoRA adapter in S3

**Already on.** `s3.enabled` is `true` in the config, so the last stage uploads
before the pod goes away:

```
[9/9] upload               OK       12s
      6 files, 84.3 MB -> s3://enlibra/dss/dev/kd/runs/<run-id>
```

| | |
|---|---|
| `final_adapter/` | **the LoRA adapter** — what you trained, tens of MB |
| `report.html` | the readable summary |
| `metrics.json`, `arena.json` | the numbers, accuracy and Elo included |
| `run.log`, `events.jsonl` | everything the terminal showed |
| `config.resolved.yaml`, `manifest.json` | exactly what produced it |

This needs **`s3:PutObject`** on `dss/dev/kd/*` — a *different* permission from
reading the teacher. If the upload stage fails, that is why, and nothing is lost:
the run is still on the pod's disk.

**Send it somewhere else:**

```bash
./run.sh train --set s3.prefix=dss/dev/my-experiment
```

**Turn it off:**

```bash
./run.sh train --set s3.enabled=false
```

**Upload by hand afterwards**, if the stage failed — on the pod, before you
terminate it:

```bash
aws s3 cp --recursive /workspace/runs/<run-id> \
    s3://enlibra/dss/dev/kd/runs/<run-id>/
```

**Copy it to your Mac instead of S3** — run this on your Mac, using the SSH
details from the RunPod console:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 -r \
    root@<pod-ip>:/workspace/runs/<run-id> ./runs/
```

**Use it later:**

```bash
aws s3 cp --recursive s3://enlibra/dss/dev/kd/runs/<run-id>/final_adapter ./my-adapter
./run.sh ask "What are stars formed from?" --adapter ./my-adapter
```

---

## Running a different YAML

The script uses three configs, because it does three different things:

| | | |
|---|---|---|
| `configs/enlibraQ3-8B.yaml` | `--config` | the **real** run, on a GPU |
| `configs/enlibraQ3-8B-smoke.yaml` | `--smoke-config` | the **rehearsal**, minutes |
| `configs/enlibraQ3-8B-mac.yaml` | `--full-config` | **all the data**, locally, hours |

Three ways to point somewhere else, in increasing order of permanence:

```bash
# 1. just this once
./run.sh --config configs/mine.yaml

# 2. just this shell
export KD_CONFIG=configs/mine.yaml
export KD_SMOKE_CONFIG=configs/mine-smoke.yaml
export KD_FULL_CONFIG=configs/mine-mac.yaml
./run.sh

# 3. from now on — edit the three lines under
#    "WHICH YAML DOES THIS RUN?" at the top of run.sh
```

`--config`, `--smoke-config` and `--full-config` go **before** the subcommand:

```bash
./run.sh --config configs/mine.yaml train
./run.sh --smoke-config configs/mine-smoke.yaml smoke
```

Putting `--config` after the subcommand also works — it is handed to `kd`, which
takes the last one it sees:

```bash
./run.sh train --config configs/mine.yaml       # same result
```

A path that does not exist stops immediately and lists what does, so a typo costs
a second rather than turning up after the install.

> **If you write your own profile, write both halves.** A "smoke test" that runs
> the real config on a laptop is not a smoke test — it is the expensive run on
> the wrong machine. Copy `configs/enlibraQ3-8B-smoke.yaml`; it is short, and it
> only overrides the models, the sizes and the limits.

## Doing one thing at a time

`./run.sh` on its own is the whole sequence. These run a single piece of it:

```bash
./run.sh doctor      # what can this machine do, and what can it reach
./run.sh check       # resolve the config and print the plan, run nothing
./run.sh smoke       # the smoke test — minutes, proves the plumbing
./run.sh full        # all the data + the whole evaluation, locally — hours
./run.sh train       # the real run, on a GPU
./run.sh setup       # install only
./run.sh ask "..."   # ask the model something
./run.sh help
```

They all accept `--config`:

```bash
./run.sh --config configs/mine.yaml doctor
./run.sh --config configs/mine.yaml check
```

Anything else is passed straight through, so `./run.sh arena --limit 20` and
`./run.sh evaluate` work without the script needing to know they exist.

---

## When something goes wrong

| It says | What to do |
|---|---|
| `models.teacher FAIL ... access denied` | A permissions problem, not a key problem. New keys will not help — ask your AWS administrator. |
| `PutObject ... is not authorized` at the `upload` stage | A **different** permission from reading the teacher: the run needs `s3:PutObject` on `dss/dev/kd/*`. Nothing is lost — the bundle is still on disk. |
| The run swaps / a step takes minutes on a Mac | The models do not fit. See [the memory table](#can-i-force-the-real-8b-teacher-on-the-mac). |
| `No AWS credentials in this shell` | Run the three `export` lines above, then run the script again. |
| `You are not inside tmux` | Run `tmux new -s kd`, then the script again. |
| `REFUSED at smoke` | Working as designed: the run cannot finish inside its limits. The message says which number to change. **Nothing was spent.** |
| `torch cannot see the GPU` | Wrong pod template. Use a **PyTorch** template. |
| `This looks like a rented machine, but no GPU is visible` | Exactly what it says — you are paying for a machine whose GPU is not there. Check `nvidia-smi`, and redeploy on a **PyTorch** template. |
| `Nothing is attached to answer this question` | You piped input, or ran it from a script. Set `KD_YES=1` to accept the confirmations in advance. |
| `uv installed but is not on PATH` | Open a new terminal and run the script again. |
| `No such config: ...` | The path after `--config` is wrong. The message lists the ones that exist. |
| Run died when SSH dropped | Use `tmux`. |

## Where to read more

- [CONFIG.md](CONFIG.md) — every setting, key by key
- [RUNPOD.md](RUNPOD.md) — the rented-GPU path in depth, and running it by hand
- [../README.md](../README.md) — what this project does and why
