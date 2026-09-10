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
| **Your Mac** (no GPU) | Installs everything, checks it, runs the **smoke test** — small stand-in models, the whole pipeline, free |
| **A RunPod pod** (GPU) | Installs everything, checks it, runs the **real training** |

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
  curriculum-sft             937  (85.6%)
  curriculum-rl              143  (13.1%)
  identity                    15  ( 1.4%)
  Train split     : 1047
  Validation split: 48
```

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

## Part 2 — On RunPod (paid)

### Start a pod

In the RunPod console:

1. **Deploy** → a GPU with **at least 48 GB** — an **L40S** or **RTX A6000**.
   A 24 GB card will not do; the two models are 20.5 GB of weights before
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

The script uses two configs, because it does two different things:

| | |
|---|---|
| `configs/enlibraQ3-8B.yaml` | the **real** run, on a GPU |
| `configs/enlibraQ3-8B-smoke.yaml` | the **rehearsal**, small enough for a laptop |

Three ways to point somewhere else, in increasing order of permanence:

```bash
# 1. just this once
./run.sh --config configs/mine.yaml

# 2. just this shell
export KD_CONFIG=configs/mine.yaml
export KD_SMOKE_CONFIG=configs/mine-smoke.yaml
./run.sh

# 3. from now on — edit the two lines under
#    "WHICH YAML DOES THIS RUN?" at the top of run.sh
```

`--config` and `--smoke-config` go **before** the subcommand:

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
./run.sh smoke       # the smoke test
./run.sh train       # the real run
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
