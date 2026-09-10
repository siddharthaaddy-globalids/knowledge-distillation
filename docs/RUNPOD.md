# Running on a rented GPU

Entirely optional. Everything in this file is off by default, nothing in the local
pipeline imports the SDKs, and a machine with no credentials runs every stage
without them.

Read this only when a run has outgrown the hardware you have.

## What you need

| | |
|---|---|
| A RunPod account and API key | https://console.runpod.io/user/settings |
| The published CUDA image | Run the **Build CUDA image** workflow (`workflow_dispatch`), which pushes `ghcr.io/<org>/kd:<sha>` |
| The optional dependencies | `uv sync --extra remote` |
| Somewhere to put the results | An S3 bucket, or accept that the bundle dies with the pod |

```bash
export RUNPOD_API_KEY=...
export AWS_ACCESS_KEY_ID=...        # if you want the results back
export AWS_SECRET_ACCESS_KEY=...
export HF_TOKEN=...                 # if the teacher is private
```

`kd doctor` reports which of these are present, without printing any of them.

## A first run that rents nothing

```bash
kd runpod gpus --config configs/finance.yaml --set runpod.enabled=true
```

This authenticates, reads the catalogue and prints what is available under your
price cap. It is the cheapest way to find out that your key works.

## Launching

```bash
kd runpod launch --config configs/finance.yaml \
  --set runpod.enabled=true \
  --set runpod.image=ghcr.io/<org>/kd:<sha> \
  --set s3.enabled=true --set s3.bucket=<bucket> \
  --set limits.max_cost_usd=2.00
```

Or put those in a profile that extends your training config, so a launch is one
argument:

```yaml
# configs/finance-pod.yaml
extends: finance.yaml

runpod:
  enabled: true
  gpu_type: RTX A4000
  max_price_per_hour: 0.60
  image: ghcr.io/<org>/kd:abc1234
s3:
  enabled: true
  bucket: my-bucket
limits:
  max_cost_usd: 2.00
  max_runtime_minutes: 90
```

```bash
kd runpod launch --config configs/finance-pod.yaml
```

## By hand, over SSH

The launcher above needs the published image. If you have not built one yet, or
you want a shell on the machine while it trains, rent a pod yourself on one of
RunPod's stock **PyTorch** templates, mount a volume at `/workspace`, and:

```bash
cd /workspace
git clone <this repo> && cd knowledge-distillation
export KD_PRICE_PER_HOUR=0.28        # the rate you agreed to
```

`scripts/runpod.sh` refuses to start unless `nvidia-smi` is present and torch can
actually see the GPU, then installs the dependencies **around** torch and hands
over to `python -m kd`. It keeps the template's CUDA build of torch rather than
fetching its own, which is the difference between ninety seconds of setup and six
minutes of paid GPU time. It writes `/workspace/kd-env.sh` so a second SSH
session is one `source` away from a working shell.

Everything it does not recognise is passed through, so it is also how you run the
checks below. `--setup-only` stops after the install; `--extra eval` adds the
benchmark group; `KD_DISPATCH_ONLY=1` prints the command it would run and exits.

### Rehearsing it before you rent anything

`--rehearse` runs this script on a machine that is not a GPU pod — your laptop, a
Mac mini — by downgrading the GPU checks to warnings. Everything else happens for
real: the arguments are parsed, the dependency set is read out of
`pyproject.toml`, the environment and `kd-env.sh` are written, and the pipeline is
handed to.

```bash
./scripts/runpod.sh --rehearse doctor
./scripts/runpod.sh --rehearse --config configs/enlibraQ3-8B-smoke.yaml
```

Without `/workspace`, it falls back to `~/kd-workspace` and says so.

Be clear about what this does and does not settle. It answers *does this script
work, are the dependencies resolvable, does the config resolve, can it reach S3*.
It answers nothing about whether the models fit or how fast a step is — different
hardware, different backend, often different model sizes. Those are the
questions the pipeline's own `smoke` stage asks on the pod, where it measures
s/step and refuses a run that cannot finish inside its limits.

### The first run, in order

Four commands, cheapest first. Each one rules out a different way the expensive
run can fail, and none of them is worth skipping on a machine you are paying for
by the second.

**1. Does the machine work?**

```bash
./scripts/runpod.sh doctor
```

Installs the dependencies, then reports the GPU, the torch build, and which
credentials are visible - `HF_TOKEN` for a private teacher, the AWS pair if the
results are meant to reach S3. It prints which are present, never their values.
A missing token found here costs nothing; found in the upload stage it costs the
whole run.

**2. Does the config fit this GPU?**

```bash
./scripts/runpod.sh check --config configs/finance.yaml
```

Resolves the config - every `extends`, every `--set`, every environment override -
and prints what the run would actually use against the hardware it found. Nothing
is loaded and nothing is trained. This is where a batch size that will not fit in
24 GB is supposed to be noticed.

**3. Does the pipeline work end to end?**

```bash
./scripts/runpod.sh --config configs/smoke.yaml
```

The full gated pipeline on tiny pools and two steps, in a couple of minutes. It
proves the stages run in order, the teacher check passes, a bundle is written, and
- with `s3.enabled` - that the upload credentials really work. The teacher-check
stage runs here too, deliberately: a smoke test that skips the one check catching
a broken teacher is not testing the thing most likely to be wrong.

Two minutes on a $0.28/hr card is about one cent. A misconfiguration found in the
fortieth minute of a real run is not.

**4. The real run.**

```bash
./scripts/runpod.sh --config configs/finance.yaml \
  --set limits.max_cost_usd=2.00 --set limits.max_runtime_minutes=90
```

Before training starts the `smoke` stage measures s/step against those limits and
refuses a run that cannot finish inside them. The common case should be "never
started", not "killed at 73%".

Watch it from a second SSH session:

```bash
tail -f /workspace/runs/$(ls -t /workspace/runs | head -1)/events.jsonl
```

S3 is optional and adds nothing to a first run. Turn it on later, when you want
the bundle to survive the pod without being copied by hand:

```bash
  --set s3.enabled=true --set s3.bucket=<bucket>
```

### Getting the adapter back

The run writes its bundle to `/workspace/runs/<run-id>/`. What you actually want
off the machine is `final_adapter/` - the LoRA weights, tens of megabytes, not the
gigabytes of base model they attach to.

Find the run id, then copy it down **before terminating the pod**:

```bash
ls -t /workspace/runs | head -1
```

From your own machine, not the pod:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 -r \
  root@<pod-ip>:/workspace/runs/<run-id> ./runs/
```

That brings the whole bundle - `final_adapter/`, `metrics.json`, `report.html`,
`events.jsonl`, `config.resolved.yaml`, `manifest.json`. Take all of it rather
than the adapter alone: `manifest.json` and `config.resolved.yaml` are what let
the adapter say what produced it, and an adapter that cannot is a file you will
not trust in a month.

**The evaluation already happened.** `evaluate` and `report` are stages of the
pipeline, so the pod measured transfer and wrote the report before it finished.
Nothing needs re-running locally - open `report.html` and read `metrics.json`.
They are also non-gate stages, so a failure there is logged and the run carries
on: a broken report never destroys a good adapter.

Re-run `kd evaluate` on your own machine only when you want something the pod run
did not produce:

```bash
# benchmark tasks and generation similarity - needs `uv sync --extra eval`,
# which the pod install deliberately skips
kd evaluate --config configs/finance.yaml \
  --adapter ./runs/<run-id>/final_adapter --tasks ...
```

If `scp` is awkward - a proxied SSH connection, or a pod with no public IP - the
alternative is S3: turn it on for the run, and the bundle is in the bucket before
the pod is released, whether the run succeeded or not.

Two things the launcher does for you that this path does not:

* **`KD_PRICE_PER_HOUR`.** Export it yourself, to the rate you agreed to, or the
  in-pod cost cap is inert. The script warns when it is missing.
* **Termination.** Nothing releases a pod you started by hand. It bills until you
  stop it, from the console or with `kd runpod stop <pod-id>`.

For anything longer than the smoke run, start it inside `tmux` - a dropped SSH
session kills a foreground run and leaves the pod billing with nothing to show.
The script warns when it is about to start a pipeline outside one.

## Three rules that protect the bill

### 1. The GPU you named, or none

The launcher rents the card in `runpod.gpu_type` and no other. Silently taking
"the next one up" is how a $0.34/hr run becomes a $2.80/hr run. If it is
unavailable it stops **before renting anything**, shows what is available under
your cap, and waits:

```
==> requested  RTX A4000 (spot, cap $0.60/hr)
!!  RTX A4000: no spot capacity right now
    available now, under your $0.60/hr cap:
      1) RTX A5000            24GB  $0.28/hr  spot
      2) RTX 4090             24GB  $0.44/hr  spot
    pick 1-2, or 'q' to abort (nothing has been rented):
```

In a non-interactive session it refuses instead, and tells you how to name one
explicitly. `--yes` accepts the **cost estimate**; it never accepts a different
GPU.

### 2. It always terminates

Every exit path — success, failure, a limit, Ctrl+C, an error inside the launcher
— releases the pod. If termination itself fails you get a loud message with the
console link, because that is the one failure that keeps costing money.

`--keep-alive MIN` leaves a successful pod up for inspection. It bills until then.

### 3. The cap is enforced from your machine

Spend is tracked locally against the rate actually agreed, so it holds even if the
pod hangs, stops logging, or never starts the pipeline. The pod also enforces the
same cap itself (the rate travels as `KD_PRICE_PER_HOUR`), but that is a second
line, not the only one.

Before any of that, the `smoke` stage measures s/step and refuses a run that
cannot finish inside its limits — the common case should be "never started", not
"killed at 73%".

## What a stopped run leaves behind

A breach is a hard stop: no evaluation, no report. What survives is the last
checkpoint written by `training.save_steps`, so set that tighter when limits are
tight.

On a rented machine the bundle **and its checkpoints** are synced to S3 before the
pod is terminated — otherwise a stopped run would cost money and leave nothing.

## Getting results back

The pod writes its run bundle to `/workspace/runs/<run-id>/` and, with
`s3.enabled`, syncs it to `s3://<bucket>/<prefix>/runs/<run-id>/`:

```
config.resolved.yaml   manifest.json   run.log   events.jsonl
metrics.json           report.html     final_adapter/
```

`manifest.json` and `config.resolved.yaml` always ship, whatever `s3.upload`
says — without them the bundle cannot say what produced it.

Feeding results back into a later run is symmetrical: any of `models.teacher`,
`models.student`, `models.teacher_adapter` and `dataset.source` may be an `s3://`
URI, fetched in `preflight` and cached locally.

```bash
kd evaluate --config configs/finance.yaml \
  --adapter s3://my-bucket/kd/runs/20260908T1412Z-finance-bb0c874/final_adapter
```

## Progress while it runs

The launcher polls pod state and prints elapsed time and spend:

```
    [ 12.5 min] RUNNING      $0.06
```

It does **not** stream container logs — RunPod has no reliable log API, and the
run bundle in S3 is the better channel: `events.jsonl` carries every step, loss
and checkpoint as structured records, and survives the pod.

## Keeping it cheap

| Lever | Effect |
|---|---|
| `gkd.lmbda` | The dominant cost. `0.5 → 0.25` removes half the on-policy generation passes. `0.0` is plain off-policy KD — several times faster. |
| `gkd.max_new_tokens` | Generation cost is linear in it. |
| `spot: true` | Roughly half price. |
| Pre-baked image | Saves 4–6 min of paid GPU time per run versus installing torch at pod start. |
| `volume_gb` | The HF cache lives on the volume, so weights download once rather than once per run. |
| `--only train` | Skip stages you have already passed on this config. |

## Troubleshooting

**`RUNPOD_API_KEY is not set`** — export it; the config only ever names the
variable.

**`runpod.image is not set`** — run the Build CUDA image workflow and pass the
tag it prints.

**`no GPU type called 'RTX 4090 Ti'`** — `kd runpod gpus` lists the exact names.

**The pod started and nothing happened** — check `docker_args` in the RunPod
console against what the launcher printed as `==> command`. With `s3.enabled` the
pod's `run.log` reaches the bucket even on failure, because `upload` runs after a
failed gate.

**A pod is still running** — `kd runpod stop <pod-id>`, or the console.
