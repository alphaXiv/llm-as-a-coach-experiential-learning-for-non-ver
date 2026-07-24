# Reproduction: LLM-as-a-Coach — Experiential Learning for Non-Verifiable Tasks (arXiv 2607.18110)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/llm-as-a-coach-experiential-learning-for-non-ver/blob/main/notebooks/reproduction.py)

**Verdict: partially reproduced** (downscaled). The paper's central claim is that preserving a judge's *textual* feedback as transferable experience and distilling the experience-conditioned teacher distribution (Experiential Learning, EL) generalizes better than compressing the same feedback into a scalar reward optimized with GRPO. In a matched 64-run comparison at 1.5B scale on Kubernetes (1× NVIDIA RTX PRO 6000 Blackwell per run, peak 16 concurrent, ≈4.2 h total):

- **Headline EL > scalar-RL claim — this run did not show the reported effect.** Paper: EL 80.0 vs RL 79.2 (WildChat test score), 40.0% vs 37.3% (AlpacaEval win rate). Here: GRPO **beat** EL on held-out in-domain (win rate vs base 0.589 vs 0.524), AlpacaEval (0.598 vs 0.540), and Dolly creative writing (0.625 vs 0.601), paired over 9 seeds (t = 4.4 / 5.4 / 3.2), robust across a 5× LR sweep and 2× training length.
- **Reward-hacking / generalization-decay claim — this run did not show the reported effect.** Doubling training doubled GRPO's train-score gain *and improved* its OOD win rates; EL stayed flat.
- **Feedback-bandwidth claim — partially aligned.** A 1-of-10 quantized directive (~3.3 bits, matching a scalar's bandwidth) produced ~no held-out gain (0.501), clearly below all rich-text feedback arms — but extracted "experience" did not separate from full-critique or rubrics-only contexts.

**Downscaling/substitutions:** Qwen2.5-1.5B-Instruct + LoRA r=32 (paper: 7–8B full FT), 512 no_robots open-ended prompts (paper: 7,500 WildChat-IF), 64–128 steps (paper: 3 epochs at batch 256×8), frozen Qwen2.5-7B coach + held-out Qwen3-8B eval judge (paper: policy itself or GPT-4o). The likely reason EL underperforms here: the distillation target is the frozen *initial 1.5B policy* conditioned on advice — a teacher that small may not exploit textual guidance.

📄 **[Detailed report with figures](reports/el-vs-scalar-rl/report.md)** · 📓 **[Self-contained marimo notebook](notebooks/reproduction.py)** (all 64 runs embedded — no downloads needed) · Compute: user-configured Kubernetes cluster via `orx exp run --backend k8s`.

## Experiment log

All experiments run the identical fixed command **`bash run.sh`** (verbatim from `orx exp status`); arms and seeds differ only by config-only commits on their branches. Each run: 1 GPU, 0.75–0.82 h (64 steps) or 1.38–1.46 h (128 steps). Only seed-0 branches are linked; seeds 1–8 are sibling branches with identical diffs apart from `SEED`.

| branch / experiment | purpose or change | exact run command | outcome (win rate vs base: heldout / alpaca / dolly) | compute |
|---|---|---|---|---|
| [`main`](../../tree/main) | publication surface | Not run as an experiment (publication surface) | — | — |
| [`orx/baseline-el-vs-rl-pipeline-base-model-reference`](../../tree/orx/baseline-el-vs-rl-pipeline-base-model-reference) | pipeline + smoke of all arms + base-model reference eval & judge diagnostics | `bash run.sh` | noise floor 0.461/0.532/0.523; judge position-A bias ~2:1 both orders | 1 GPU, 0.17 h |
| [`orx/el-seed0`](../../tree/orx/el-seed0) (+ seeds 1–8) | EL: experience-context reverse-KL distillation | `bash run.sh` | 0.524/0.540/0.601 (9-seed mean) | 9×1 GPU, ~0.8 h each |
| [`orx/grpo-seed0`](../../tree/orx/grpo-seed0) (+ seeds 1–8) | scalar-rubric GRPO baseline | `bash run.sh` | **0.589/0.598/0.625** (9-seed mean) | 9×1 GPU, ~0.8 h each |
| [`orx/full-critique-seed0`](../../tree/orx/full-critique-seed0) (+ 1–5) | control: whole critique as teacher context | `bash run.sh` | 0.536/0.532/0.592 (6-seed mean) | 6×1 GPU, ~0.8 h each |
| [`orx/rubrics-only-seed0`](../../tree/orx/rubrics-only-seed0) (+ 1–5) | control: rubric text as teacher context | `bash run.sh` | 0.533/0.535/0.618 (6-seed mean) | 6×1 GPU, ~0.8 h each |
| [`orx/directive-seed0`](../../tree/orx/directive-seed0) (+ 1–8) | control: 3.3-bit quantized directive context | `bash run.sh` | 0.501/0.497/0.567 (9-seed mean) — ≈ noise floor in-domain | 9×1 GPU, ~0.8 h each |
| [`orx/el-long-seed0`](../../tree/orx/el-long-seed0) (+ 1–5) | claim 2: EL at 128 steps (4 epochs) | `bash run.sh` | 0.523/0.527/0.618 — flat vs 64 steps | 6×1 GPU, ~1.4 h each |
| [`orx/grpo-long-seed0`](../../tree/orx/grpo-long-seed0) (+ 1–5) | claim 2: GRPO at 128 steps | `bash run.sh` | **0.617/0.607/0.688** — improves with steps | 6×1 GPU, ~1.4 h each |
| [`orx/el-lr2e-5-seed0`](../../tree/orx/el-lr2e-5-seed0) / [`orx/el-lr5e-5-seed0`](../../tree/orx/el-lr5e-5-seed0) (+ seeds 1–2) | robustness: EL at 2×/5× LR | `bash run.sh` | 0.527 / 0.508 heldout — LR does not rescue EL | 6×1 GPU, ~0.8 h each |
| [`orx/grpo-lr2e-5-seed0`](../../tree/orx/grpo-lr2e-5-seed0) / [`orx/grpo-lr5e-5-seed0`](../../tree/orx/grpo-lr5e-5-seed0) (+ seeds 1–2) | matched GRPO LR controls | `bash run.sh` | 0.606 / **0.692** heldout — GRPO scales with LR | 6×1 GPU, ~0.8 h each |

One additional launch failed before any science ran (k8s manifest quoting; fixed in [`5713f2f`](../../commit/5713f2f)). A duplicate node `orx/el-long-seed0-2` was never run.

## Repository layout

- `reports/el-vs-scalar-rl/report.md` — the detailed tutorial-style report (figures in `images/`).
- `notebooks/reproduction.py` — marimo notebook, self-contained (embedded results); `marimo edit notebooks/reproduction.py` locally or use the Molab badge above.
- `config.py`, `main.py`, `run.sh`, `src/`, `.orx/k8s.yaml` — the complete training/eval pipeline (baseline defaults: `ARM = "baseline"`; arm branches change only `ARM`/`SEED`/`TRAIN_STEPS`/`LR`).
- `autoresearch.json` — machine-readable reproduction metadata.

## Method sketch

One pipeline, five arms. Each training step samples 4 responses for each of 16 prompts from the policy, gets one coach call per response (critique + 1–10 score + `<experience>` extraction), then applies the arm's update: GRPO uses only the score (group-normalized REINFORCE); the four context arms condition a frozen copy of the initial policy on their context (experience / full critique / rubrics / directive) and minimize token-level reverse KL to it over the student's top-256 tokens. Evaluation generates responses on three held-out sets from the trained policy and the frozen base, and asks Qwen3-8B to judge pairs in both presentation orders (the baseline run showed a 2:1 position bias, cancelled by order-averaging).
