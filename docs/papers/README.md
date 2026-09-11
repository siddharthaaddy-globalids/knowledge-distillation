# Papers behind the `gkd` settings

The PDFs in this folder are the sources for the values in `configs/*.yaml` under
`gkd:` and for the "How it was trained" section of every report. Each entry says
what the paper found that bears on a knob in this repository; the numbers are
the papers' own.

| File | Paper | What it settles here |
|---|---|---|
| `gkd-agarwal-2024-on-policy-distillation.pdf` | Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*, ICLR 2024 ([arXiv 2306.13649](https://arxiv.org/abs/2306.13649)) | The method TRL's `GKDTrainer` implements: `beta` (JSD interpolation), `lmbda` (student data fraction), `seq_kd`. The divergence and data-fraction sweeps. |
| `distillm-ko-2024-skew-kl.pdf` | Ko et al., *DistiLLM: Towards Streamlined Distillation for Large Language Models*, ICML 2024 ([arXiv 2402.03898](https://arxiv.org/abs/2402.03898)) | Why a mild skew (α = 0.1) beats JSD(0.5); that a fixed 50% on-policy mix is too much too early, and an adaptive ramp from 0 works better and 2–3x faster. Includes a LoRA result on OpenLLaMA-7B → 3B. |
| `minillm-gu-2024-reverse-kl.pdf` | Gu et al., *MiniLLM: Knowledge Distillation of Large Language Models*, ICLR 2024 ([arXiv 2306.08543](https://arxiv.org/abs/2306.08543)) | Reverse KL on student-generated text for instruction following, GPT-2 → LLaMA scale. The mode-seeking argument for small students. |

Not a paper, but the most recent practitioner recipe, on Qwen3 with LoRA:
Thinking Machines Lab, *On-Policy Distillation* (2025),
<https://thinkingmachines.ai/blog/on-policy-distillation/> — per-token reverse
KL on student rollouts, LoRA rank 128, 4 samples per prompt, temperature 1.0;
reaches the teacher 7–10x faster than RL in gradient steps.

## What the papers say, knob by knob

**`beta` — which divergence.** GKD's headline is that it is task-dependent,
but the pattern is consistent (Agarwal Fig. 4, 6, 7, 10, A.12–A.14):

- With **greedy decoding at eval**, the divergence barely matters
  (XSum, T5-small, λ=1: forward KL 16.4, JSD(0.1) 16.6, JSD(0.5) 16.6,
  JSD(0.9) 16.3, reverse KL 15.6 ROUGE-2).
- With **sampling at eval**, mode-seeking wins clearly (same setup, γ=1:
  forward KL 13.2, JSD(0.5) 15.5, JSD(0.9) 15.2, reverse KL 14.5).
- **Translation**: JSD(0.1) best on T5-small (0.85 BLEU gain vs 0.55 forward
  KL); differences shrink as the student grows.
- **GSM8K reasoning**: forward KL and reverse KL both good (8.8 and 8.0 points
  on T5-base), the middle JSDs slightly worse.
- **Instruction tuning (FLAN)**: reverse KL far ahead of forward KL
  (+1.9 vs −0.4 MMLU points). Their explanation: mode-seeking makes the
  student "zero in on the main intent" of an instruction.
- Their RL note: "we recommend using reverse KL or JSD(0.9)" when the
  student must stay close to a reference — which is also the distillation case
  where the student is much smaller than the teacher.

DistiLLM goes further: JSD's two skew terms cannot both be mild at once, and a
single skew KL with **α = 0.1** (skew KL or skew reverse KL) beats KL, reverse
KL and JSD(0.9) on every instruction-following eval in Table 1 (Dolly 25.21 vs
JSD 24.34 ROUGE-L), converging faster (Fig. 6). Performance falls off sharply
above α ≈ 0.3 for skew reverse KL (Fig. 8). TRL has no skew-KL option, so the
nearest available setting is `beta` near 0.9 — reverse-leaning JSD.

**`lmbda` — how much on-policy.** Every ablation in GKD has λ ≥ 0.5 beating
λ = 0, and λ = 1 usually best (Agarwal Fig. 6, 7, A.12–A.14): GSM8K T5-base
forward KL 4.7 → 6.8 → 8.8 for λ = 0, 0.5, 1; WMT T5-small JSD(0.1)
0.28 → 0.71 → 0.85. Fig. 8: "performance consistently improves as the
proportion of on-policy data increases, provided that at least 25% of the data
is on-policy." Cost is 1.8–2.2x the compute of off-policy (A.2).

DistiLLM's qualification (Fig. 1, Tab. 2, 5): early student generations are bad
enough to mislead the teacher, so start at φ = 0 and ramp up as validation
loss stops improving; the ramp ends near **0.3–0.4**, matches or beats a fixed
50% mix, and runs 2.2–3.4x faster. Fixed mixes that were tuned by hand landed
in the same 0.3–0.4 range.

**`temperature` — student sampling.** GKD trains on-policy at **γ = 1.0**
("to encourage diversity in student generated sequences", §3.1), and the
Thinking Machines recipe also samples at 1.0. Lower values reduce the
diversity that is the point of on-policy data. Only matters when `lmbda > 0`.

**`seq_kd` — teacher-written data.** SeqKD trails on-policy GKD everywhere it
is compared (Agarwal Fig. 1, 9) and costs a teacher generation per sample.
Keep it `false`; a curriculum written by hand is better data than the teacher's
rewrite of it.

**Learning rate and LoRA.** GKD used **3e-4** (T5-base/large; 1e-3 for the
77M model) with 2k warmup and linear cooldown, batch 32, and noted reverse KL
"was more sensitive to higher LRs" (A.3). DistiLLM's 7B → 3B result used LoRA
without stating the rank; Thinking Machines used rank **128** and found LoRA
trails full fine-tuning by only 6% after on-policy distillation versus 13%
after SFT.

## What this means for `configs/enlibraQ25-3B.yaml`

| Knob | Now | Literature | Suggested next |
|---|---|---|---|
| `beta` | 0.5 | Reverse-leaning wins for small students and instruction-shaped data | **0.9** |
| `lmbda` | 0.0 | ≥ 0.25 for any on-policy gain; ramp rather than jump | 0.0 → **0.25–0.5** once the `<think>` rollout issue is handled |
| `temperature` | 0.7 | 1.0 in every on-policy recipe | **1.0** when `lmbda > 0` |
| `seq_kd` | false | Never beats on-policy; costs a teacher generation | keep `false` |
| `learning_rate` | 3e-4 | 3e-4 is the papers' default; reverse KL likes it no higher | keep, or 2e-4 with `beta` 0.9 |
| `lora.r` | 32 | 128 in the one LoRA-specific recipe | 64–128 if memory allows |

Off-policy at `beta: 0.5` is the setting every paper uses as the baseline to
beat. It is the right first run — cheap and stable — and the report's
"How it was trained" section says so; the table above is the order to change
things in afterwards.
