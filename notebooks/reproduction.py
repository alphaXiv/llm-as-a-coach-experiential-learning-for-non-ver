# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "matplotlib", "numpy"]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Does textual "experience" beat a scalar reward? A downscaled reproduction of *LLM-as-a-Coach* (arXiv 2607.18110)

    The paper claims that for open-ended, non-verifiable tasks, converting judge feedback into
    **transferable textual experience** and distilling the experience-conditioned teacher
    distribution into the policy (**Experiential Learning, EL**) generalizes better than
    compressing the same feedback into a **scalar reward** optimized with GRPO.

    This notebook is the self-contained evidence archive of a 64-run reproduction on a
    16-GPU Kubernetes cluster (NVIDIA RTX PRO 6000 Blackwell). Everything below renders from
    data embedded in this file — no external downloads needed.

    **Setup (downscaled, matched across arms):** Qwen2.5-1.5B-Instruct policy + LoRA r=32,
    frozen Qwen2.5-7B-Instruct coach/judge for training feedback, held-out Qwen3-8B judge for
    evaluation (order-swapped pairwise vs the frozen base policy), 512 open-ended training
    prompts (no_robots), evaluated on 128 held-out in-domain prompts + AlpacaEval (100) +
    Dolly creative-writing (64) as unseen OOD sets.
    """)
    return


@app.cell
def _():
    RUNS = [{"group": "baseline", "seed": 0, "steps": 0, "heldout_wr": 0.4609, "heldout_abs": 7.1953, "alpaca_wr": 0.5325, "alpaca_abs": 7.53, "dolly_wr": 0.5234, "dolly_abs": 7.4688, "heldout_abs_base": 7.3828, "train_first8": None, "train_last8": None}, {"group": "directive", "seed": 0, "steps": 64, "heldout_wr": 0.502, "heldout_abs": 7.3672, "alpaca_wr": 0.5525, "alpaca_abs": 7.78, "dolly_wr": 0.6133, "dolly_abs": 7.7969, "heldout_abs_base": 7.3984, "train_first8": 7.084, "train_last8": 7.09}, {"group": "directive", "seed": 1, "steps": 64, "heldout_wr": 0.4863, "heldout_abs": 7.3594, "alpaca_wr": 0.4175, "alpaca_abs": 7.59, "dolly_wr": 0.6016, "dolly_abs": 8.0469, "heldout_abs_base": 7.3828, "train_first8": 6.818, "train_last8": 7.078}, {"group": "directive", "seed": 2, "steps": 64, "heldout_wr": 0.4902, "heldout_abs": 7.4766, "alpaca_wr": 0.4825, "alpaca_abs": 7.73, "dolly_wr": 0.5703, "dolly_abs": 8.0312, "heldout_abs_base": 7.3203, "train_first8": 6.961, "train_last8": 7.098}, {"group": "directive", "seed": 3, "steps": 64, "heldout_wr": 0.5117, "heldout_abs": 7.5703, "alpaca_wr": 0.5525, "alpaca_abs": 7.74, "dolly_wr": 0.543, "dolly_abs": 7.875, "heldout_abs_base": 7.4453, "train_first8": 6.975, "train_last8": 7.068}, {"group": "directive", "seed": 4, "steps": 64, "heldout_wr": 0.5215, "heldout_abs": 7.5, "alpaca_wr": 0.455, "alpaca_abs": 7.86, "dolly_wr": 0.5703, "dolly_abs": 7.9531, "heldout_abs_base": 7.3359, "train_first8": 6.979, "train_last8": 7.105}, {"group": "directive", "seed": 5, "steps": 64, "heldout_wr": 0.4883, "heldout_abs": 7.2266, "alpaca_wr": 0.48, "alpaca_abs": 7.65, "dolly_wr": 0.5234, "dolly_abs": 7.8594, "heldout_abs_base": 7.4141, "train_first8": 7.033, "train_last8": 7.133}, {"group": "directive", "seed": 6, "steps": 64, "heldout_wr": 0.5039, "heldout_abs": 7.4453, "alpaca_wr": 0.51, "alpaca_abs": 7.8, "dolly_wr": 0.5742, "dolly_abs": 7.9688, "heldout_abs_base": 7.1328, "train_first8": 6.844, "train_last8": 7.303}, {"group": "directive", "seed": 7, "steps": 64, "heldout_wr": 0.4922, "heldout_abs": 7.5156, "alpaca_wr": 0.445, "alpaca_abs": 7.88, "dolly_wr": 0.5586, "dolly_abs": 7.9062, "heldout_abs_base": 7.2266, "train_first8": 7.078, "train_last8": 6.922}, {"group": "directive", "seed": 8, "steps": 64, "heldout_wr": 0.5117, "heldout_abs": 7.5625, "alpaca_wr": 0.5825, "alpaca_abs": 7.8, "dolly_wr": 0.5508, "dolly_abs": 8.1719, "heldout_abs_base": 7.25, "train_first8": 6.82, "train_last8": 7.199}, {"group": "el_long", "seed": 0, "steps": 128, "heldout_wr": 0.4941, "heldout_abs": 7.5078, "alpaca_wr": 0.5825, "alpaca_abs": 7.9, "dolly_wr": 0.6445, "dolly_abs": 7.8906, "heldout_abs_base": 7.3594, "train_first8": 7.033, "train_last8": 7.168}, {"group": "el_long", "seed": 1, "steps": 128, "heldout_wr": 0.5293, "heldout_abs": 7.5469, "alpaca_wr": 0.4825, "alpaca_abs": 7.92, "dolly_wr": 0.6094, "dolly_abs": 7.9531, "heldout_abs_base": 7.3672, "train_first8": 6.768, "train_last8": 7.062}, {"group": "el_long", "seed": 2, "steps": 128, "heldout_wr": 0.5254, "heldout_abs": 7.5469, "alpaca_wr": 0.56, "alpaca_abs": 7.84, "dolly_wr": 0.5859, "dolly_abs": 8.1406, "heldout_abs_base": 7.3281, "train_first8": 7.051, "train_last8": 7.221}, {"group": "el_long", "seed": 3, "steps": 128, "heldout_wr": 0.5098, "heldout_abs": 7.4375, "alpaca_wr": 0.555, "alpaca_abs": 7.75, "dolly_wr": 0.6172, "dolly_abs": 8.0469, "heldout_abs_base": 7.4219, "train_first8": 6.979, "train_last8": 7.32}, {"group": "el_long", "seed": 4, "steps": 128, "heldout_wr": 0.5391, "heldout_abs": 7.5859, "alpaca_wr": 0.495, "alpaca_abs": 7.72, "dolly_wr": 0.6445, "dolly_abs": 8.1875, "heldout_abs_base": 7.3047, "train_first8": 6.914, "train_last8": 7.238}, {"group": "el_long", "seed": 5, "steps": 128, "heldout_wr": 0.541, "heldout_abs": 7.5234, "alpaca_wr": 0.49, "alpaca_abs": 7.67, "dolly_wr": 0.6055, "dolly_abs": 7.9375, "heldout_abs_base": 7.4141, "train_first8": 6.963, "train_last8": 7.199}, {"group": "el_lr2e5", "seed": 0, "steps": 64, "heldout_wr": 0.5312, "heldout_abs": 7.6328, "alpaca_wr": 0.585, "alpaca_abs": 7.9, "dolly_wr": 0.625, "dolly_abs": 8.0625, "heldout_abs_base": 7.375, "train_first8": 7.037, "train_last8": 7.141}, {"group": "el_lr2e5", "seed": 1, "steps": 64, "heldout_wr": 0.5371, "heldout_abs": 7.5625, "alpaca_wr": 0.4475, "alpaca_abs": 7.64, "dolly_wr": 0.6562, "dolly_abs": 8.1562, "heldout_abs_base": 7.3828, "train_first8": 6.936, "train_last8": 7.043}, {"group": "el_lr2e5", "seed": 2, "steps": 64, "heldout_wr": 0.5117, "heldout_abs": 7.5, "alpaca_wr": 0.535, "alpaca_abs": 7.66, "dolly_wr": 0.5898, "dolly_abs": 8.0781, "heldout_abs_base": 7.3359, "train_first8": 7.029, "train_last8": 7.115}, {"group": "el_lr5e5", "seed": 0, "steps": 64, "heldout_wr": 0.5, "heldout_abs": 7.5625, "alpaca_wr": 0.5625, "alpaca_abs": 7.95, "dolly_wr": 0.5625, "dolly_abs": 7.7969, "heldout_abs_base": 7.3672, "train_first8": 7.148, "train_last8": 7.164}, {"group": "el_lr5e5", "seed": 1, "steps": 64, "heldout_wr": 0.5078, "heldout_abs": 7.5547, "alpaca_wr": 0.5775, "alpaca_abs": 7.71, "dolly_wr": 0.5859, "dolly_abs": 8.0469, "heldout_abs_base": 7.3984, "train_first8": 6.996, "train_last8": 7.027}, {"group": "el_lr5e5", "seed": 2, "steps": 64, "heldout_wr": 0.5156, "heldout_abs": 7.6172, "alpaca_wr": 0.5375, "alpaca_abs": 7.59, "dolly_wr": 0.5977, "dolly_abs": 8.125, "heldout_abs_base": 7.3203, "train_first8": 7.176, "train_last8": 7.135}, {"group": "el", "seed": 0, "steps": 64, "heldout_wr": 0.5293, "heldout_abs": 7.7109, "alpaca_wr": 0.5825, "alpaca_abs": 7.72, "dolly_wr": 0.6406, "dolly_abs": 8.0, "heldout_abs_base": 7.3594, "train_first8": 7.064, "train_last8": 7.143}, {"group": "el", "seed": 1, "steps": 64, "heldout_wr": 0.5527, "heldout_abs": 7.5703, "alpaca_wr": 0.605, "alpaca_abs": 7.83, "dolly_wr": 0.6367, "dolly_abs": 8.0625, "heldout_abs_base": 7.3828, "train_first8": 6.836, "train_last8": 7.102}, {"group": "el", "seed": 2, "steps": 64, "heldout_wr": 0.5352, "heldout_abs": 7.6875, "alpaca_wr": 0.515, "alpaca_abs": 7.78, "dolly_wr": 0.625, "dolly_abs": 8.1406, "heldout_abs_base": 7.3516, "train_first8": 7.033, "train_last8": 7.102}, {"group": "el", "seed": 3, "steps": 64, "heldout_wr": 0.4941, "heldout_abs": 7.5156, "alpaca_wr": 0.5275, "alpaca_abs": 7.81, "dolly_wr": 0.5938, "dolly_abs": 7.9219, "heldout_abs_base": 7.4219, "train_first8": 6.9, "train_last8": 7.082}, {"group": "el", "seed": 4, "steps": 64, "heldout_wr": 0.5254, "heldout_abs": 7.5781, "alpaca_wr": 0.555, "alpaca_abs": 7.85, "dolly_wr": 0.5977, "dolly_abs": 8.1562, "heldout_abs_base": 7.3203, "train_first8": 7.014, "train_last8": 7.207}, {"group": "el", "seed": 5, "steps": 64, "heldout_wr": 0.5137, "heldout_abs": 7.4922, "alpaca_wr": 0.525, "alpaca_abs": 7.8, "dolly_wr": 0.6016, "dolly_abs": 7.9531, "heldout_abs_base": 7.4219, "train_first8": 7.008, "train_last8": 7.238}, {"group": "el", "seed": 6, "steps": 64, "heldout_wr": 0.498, "heldout_abs": 7.4531, "alpaca_wr": 0.53, "alpaca_abs": 7.86, "dolly_wr": 0.5039, "dolly_abs": 8.0469, "heldout_abs_base": 7.1406, "train_first8": 6.848, "train_last8": 7.342}, {"group": "el", "seed": 7, "steps": 64, "heldout_wr": 0.5449, "heldout_abs": 7.5312, "alpaca_wr": 0.515, "alpaca_abs": 7.93, "dolly_wr": 0.5977, "dolly_abs": 7.9688, "heldout_abs_base": 7.2734, "train_first8": 7.02, "train_last8": 7.033}, {"group": "el", "seed": 8, "steps": 64, "heldout_wr": 0.5215, "heldout_abs": 7.5, "alpaca_wr": 0.505, "alpaca_abs": 7.91, "dolly_wr": 0.6133, "dolly_abs": 8.0156, "heldout_abs_base": 7.3281, "train_first8": 6.99, "train_last8": 7.271}, {"group": "full_critique", "seed": 0, "steps": 64, "heldout_wr": 0.4824, "heldout_abs": 7.5469, "alpaca_wr": 0.585, "alpaca_abs": 7.7778, "dolly_wr": 0.582, "dolly_abs": 7.875, "heldout_abs_base": 7.375, "train_first8": 7.053, "train_last8": 7.223}, {"group": "full_critique", "seed": 1, "steps": 64, "heldout_wr": 0.5176, "heldout_abs": 7.5391, "alpaca_wr": 0.495, "alpaca_abs": 7.69, "dolly_wr": 0.6797, "dolly_abs": 8.1562, "heldout_abs_base": 7.3828, "train_first8": 6.857, "train_last8": 7.053}, {"group": "full_critique", "seed": 2, "steps": 64, "heldout_wr": 0.5586, "heldout_abs": 7.5781, "alpaca_wr": 0.5225, "alpaca_abs": 7.7273, "dolly_wr": 0.6172, "dolly_abs": 8.2656, "heldout_abs_base": 7.3438, "train_first8": 6.984, "train_last8": 7.135}, {"group": "full_critique", "seed": 3, "steps": 64, "heldout_wr": 0.5703, "heldout_abs": 7.6094, "alpaca_wr": 0.54, "alpaca_abs": 7.91, "dolly_wr": 0.543, "dolly_abs": 8.0469, "heldout_abs_base": 7.4219, "train_first8": 7.023, "train_last8": 7.158}, {"group": "full_critique", "seed": 4, "steps": 64, "heldout_wr": 0.5566, "heldout_abs": 7.5078, "alpaca_wr": 0.53, "alpaca_abs": 7.86, "dolly_wr": 0.582, "dolly_abs": 8.0938, "heldout_abs_base": 7.3438, "train_first8": 6.957, "train_last8": 7.188}, {"group": "full_critique", "seed": 5, "steps": 64, "heldout_wr": 0.5293, "heldout_abs": 7.4141, "alpaca_wr": 0.52, "alpaca_abs": 7.75, "dolly_wr": 0.5469, "dolly_abs": 8.125, "heldout_abs_base": 7.3984, "train_first8": 7.002, "train_last8": 7.229}, {"group": "grpo_long", "seed": 0, "steps": 128, "heldout_wr": 0.6191, "heldout_abs": 7.8672, "alpaca_wr": 0.65, "alpaca_abs": 7.87, "dolly_wr": 0.668, "dolly_abs": 8.1875, "heldout_abs_base": 7.3516, "train_first8": 7.066, "train_last8": 7.416}, {"group": "grpo_long", "seed": 1, "steps": 128, "heldout_wr": 0.6113, "heldout_abs": 7.7031, "alpaca_wr": 0.5575, "alpaca_abs": 7.98, "dolly_wr": 0.6641, "dolly_abs": 8.25, "heldout_abs_base": 7.3906, "train_first8": 6.771, "train_last8": 7.389}, {"group": "grpo_long", "seed": 2, "steps": 128, "heldout_wr": 0.6113, "heldout_abs": 7.8984, "alpaca_wr": 0.6525, "alpaca_abs": 7.97, "dolly_wr": 0.7148, "dolly_abs": 8.2812, "heldout_abs_base": 7.3281, "train_first8": 7.008, "train_last8": 7.471}, {"group": "grpo_long", "seed": 3, "steps": 128, "heldout_wr": 0.627, "heldout_abs": 7.8594, "alpaca_wr": 0.57, "alpaca_abs": 7.84, "dolly_wr": 0.707, "dolly_abs": 8.0781, "heldout_abs_base": 7.4297, "train_first8": 6.959, "train_last8": 7.504}, {"group": "grpo_long", "seed": 4, "steps": 128, "heldout_wr": 0.6523, "heldout_abs": 7.7812, "alpaca_wr": 0.6575, "alpaca_abs": 8.01, "dolly_wr": 0.6523, "dolly_abs": 8.2656, "heldout_abs_base": 7.3125, "train_first8": 6.957, "train_last8": 7.352}, {"group": "grpo_long", "seed": 5, "steps": 128, "heldout_wr": 0.6328, "heldout_abs": 7.7969, "alpaca_wr": 0.64, "alpaca_abs": 7.98, "dolly_wr": 0.7266, "dolly_abs": 8.1719, "heldout_abs_base": 7.4062, "train_first8": 7.033, "train_last8": 7.51}, {"group": "grpo_lr2e5", "seed": 0, "steps": 64, "heldout_wr": 0.6289, "heldout_abs": 7.6875, "alpaca_wr": 0.66, "alpaca_abs": 8.05, "dolly_wr": 0.6914, "dolly_abs": 8.0938, "heldout_abs_base": 7.3906, "train_first8": 7.037, "train_last8": 7.385}, {"group": "grpo_lr2e5", "seed": 1, "steps": 64, "heldout_wr": 0.5645, "heldout_abs": 7.6797, "alpaca_wr": 0.6275, "alpaca_abs": 7.98, "dolly_wr": 0.7227, "dolly_abs": 8.1562, "heldout_abs_base": 7.3828, "train_first8": 6.814, "train_last8": 7.307}, {"group": "grpo_lr2e5", "seed": 2, "steps": 64, "heldout_wr": 0.625, "heldout_abs": 7.8125, "alpaca_wr": 0.645, "alpaca_abs": 8.05, "dolly_wr": 0.6484, "dolly_abs": 8.25, "heldout_abs_base": 7.3281, "train_first8": 6.99, "train_last8": 7.238}, {"group": "grpo_lr5e5", "seed": 0, "steps": 64, "heldout_wr": 0.7188, "heldout_abs": 7.9609, "alpaca_wr": 0.705, "alpaca_abs": 8.15, "dolly_wr": 0.7266, "dolly_abs": 8.2031, "heldout_abs_base": 7.3828, "train_first8": 7.178, "train_last8": 7.545}, {"group": "grpo_lr5e5", "seed": 1, "steps": 64, "heldout_wr": 0.6465, "heldout_abs": 7.8984, "alpaca_wr": 0.62, "alpaca_abs": 7.9798, "dolly_wr": 0.6992, "dolly_abs": 8.2812, "heldout_abs_base": 7.3828, "train_first8": 6.891, "train_last8": 7.41}, {"group": "grpo_lr5e5", "seed": 2, "steps": 64, "heldout_wr": 0.7109, "heldout_abs": 7.9688, "alpaca_wr": 0.6925, "alpaca_abs": 8.1, "dolly_wr": 0.7539, "dolly_abs": 8.2812, "heldout_abs_base": 7.3281, "train_first8": 7.129, "train_last8": 7.535}, {"group": "grpo", "seed": 0, "steps": 64, "heldout_wr": 0.5977, "heldout_abs": 7.6172, "alpaca_wr": 0.635, "alpaca_abs": 7.87, "dolly_wr": 0.6406, "dolly_abs": 8.0781, "heldout_abs_base": 7.4375, "train_first8": 7.055, "train_last8": 7.254}, {"group": "grpo", "seed": 1, "steps": 64, "heldout_wr": 0.5508, "heldout_abs": 7.5781, "alpaca_wr": 0.6, "alpaca_abs": 7.91, "dolly_wr": 0.6953, "dolly_abs": 8.0312, "heldout_abs_base": 7.3906, "train_first8": 6.701, "train_last8": 7.187}, {"group": "grpo", "seed": 2, "steps": 64, "heldout_wr": 0.5918, "heldout_abs": 7.6484, "alpaca_wr": 0.58, "alpaca_abs": 8.18, "dolly_wr": 0.6133, "dolly_abs": 7.9531, "heldout_abs_base": 7.3359, "train_first8": 7.012, "train_last8": 7.186}, {"group": "grpo", "seed": 3, "steps": 64, "heldout_wr": 0.5996, "heldout_abs": 7.7188, "alpaca_wr": 0.57, "alpaca_abs": 7.81, "dolly_wr": 0.6016, "dolly_abs": 8.2188, "heldout_abs_base": 7.4375, "train_first8": 6.926, "train_last8": 7.211}, {"group": "grpo", "seed": 4, "steps": 64, "heldout_wr": 0.6191, "heldout_abs": 7.8281, "alpaca_wr": 0.6525, "alpaca_abs": 8.05, "dolly_wr": 0.6328, "dolly_abs": 8.3125, "heldout_abs_base": 7.3438, "train_first8": 6.947, "train_last8": 7.219}, {"group": "grpo", "seed": 5, "steps": 64, "heldout_wr": 0.6289, "heldout_abs": 7.7031, "alpaca_wr": 0.595, "alpaca_abs": 7.88, "dolly_wr": 0.6406, "dolly_abs": 7.9844, "heldout_abs_base": 7.3984, "train_first8": 6.961, "train_last8": 7.328}, {"group": "grpo", "seed": 6, "steps": 64, "heldout_wr": 0.6074, "heldout_abs": 7.6172, "alpaca_wr": 0.61, "alpaca_abs": 7.99, "dolly_wr": 0.5312, "dolly_abs": 7.8906, "heldout_abs_base": 7.1484, "train_first8": 6.953, "train_last8": 7.307}, {"group": "grpo", "seed": 7, "steps": 64, "heldout_wr": 0.5566, "heldout_abs": 7.6953, "alpaca_wr": 0.545, "alpaca_abs": 7.92, "dolly_wr": 0.6172, "dolly_abs": 8.0312, "heldout_abs_base": 7.2656, "train_first8": 7.08, "train_last8": 7.203}, {"group": "grpo", "seed": 8, "steps": 64, "heldout_wr": 0.5508, "heldout_abs": 7.5312, "alpaca_wr": 0.5925, "alpaca_abs": 7.76, "dolly_wr": 0.6523, "dolly_abs": 8.1094, "heldout_abs_base": 7.25, "train_first8": 6.941, "train_last8": 7.328}, {"group": "rubrics_only", "seed": 0, "steps": 64, "heldout_wr": 0.5234, "heldout_abs": 7.6562, "alpaca_wr": 0.5525, "alpaca_abs": 7.93, "dolly_wr": 0.6484, "dolly_abs": 8.125, "heldout_abs_base": 7.375, "train_first8": 7.059, "train_last8": 7.281}, {"group": "rubrics_only", "seed": 1, "steps": 64, "heldout_wr": 0.5254, "heldout_abs": 7.6484, "alpaca_wr": 0.495, "alpaca_abs": 7.71, "dolly_wr": 0.6289, "dolly_abs": 8.25, "heldout_abs_base": 7.3828, "train_first8": 6.871, "train_last8": 7.131}, {"group": "rubrics_only", "seed": 2, "steps": 64, "heldout_wr": 0.5332, "heldout_abs": 7.6875, "alpaca_wr": 0.58, "alpaca_abs": 7.97, "dolly_wr": 0.6211, "dolly_abs": 8.2031, "heldout_abs_base": 7.3203, "train_first8": 6.99, "train_last8": 7.213}, {"group": "rubrics_only", "seed": 3, "steps": 64, "heldout_wr": 0.4746, "heldout_abs": 7.5859, "alpaca_wr": 0.515, "alpaca_abs": 7.88, "dolly_wr": 0.6016, "dolly_abs": 8.1406, "heldout_abs_base": 7.4375, "train_first8": 7.064, "train_last8": 7.211}, {"group": "rubrics_only", "seed": 4, "steps": 64, "heldout_wr": 0.6016, "heldout_abs": 7.7109, "alpaca_wr": 0.57, "alpaca_abs": 7.94, "dolly_wr": 0.5938, "dolly_abs": 8.1719, "heldout_abs_base": 7.3438, "train_first8": 7.021, "train_last8": 7.209}, {"group": "rubrics_only", "seed": 5, "steps": 64, "heldout_wr": 0.5391, "heldout_abs": 7.5781, "alpaca_wr": 0.495, "alpaca_abs": 7.89, "dolly_wr": 0.6133, "dolly_abs": 7.9688, "heldout_abs_base": 7.4062, "train_first8": 7.031, "train_last8": 7.197}]
    MEAN_CURVES = {"directive": [6.967, 6.967, 6.8872, 7.0781, 6.7917, 6.9045, 7.1163, 6.9254, 7.0521, 7.1146, 7.0573, 7.1528, 6.9896, 6.9358, 7.1597, 7.1927, 7.0851, 7.158, 7.1511, 7.1927, 7.1493, 7.1076, 7.2031, 7.2135, 7.092, 7.0885, 7.0643, 7.0712, 7.0243, 7.1285, 7.0746, 7.2569, 6.9722, 7.1701, 7.1892, 7.0816, 7.0608, 7.0347, 7.1563, 7.1267, 7.2118, 7.2257, 7.1528, 7.2257, 7.2951, 7.1181, 7.1892, 7.1076, 6.9184, 7.0747, 7.0434, 7.099, 7.2274, 7.1337, 7.1215, 7.1684, 7.1493, 7.1649, 7.1441, 7.0191, 7.1927, 7.059, 7.0261, 7.1302], "el_long": [6.9036, 7.0625, 6.8385, 7.0911, 6.8672, 6.961, 7.0625, 6.8229, 7.1589, 7.224, 7.1432, 7.2682, 6.9948, 7.0625, 7.2188, 7.2318, 7.0182, 7.112, 7.138, 7.1562, 7.2448, 7.0052, 7.224, 7.3985, 7.1068, 7.0495, 7.0573, 7.1432, 7.1589, 7.1641, 7.3151, 7.2656, 7.0807, 7.2786, 7.2318, 7.1276, 7.0781, 7.2161, 7.1979, 7.1823, 7.1849, 7.0937, 7.2604, 7.1719, 7.2396, 7.2317, 7.1459, 7.224, 7.0573, 7.1771, 7.224, 7.1667, 7.211, 7.1536, 7.3281, 7.2187, 7.289, 7.151, 7.2031, 7.0625, 7.2083, 6.9192, 7.2005, 7.276, 7.2813, 7.1823, 7.3438, 7.263, 7.2422, 7.2735, 7.2552, 7.0495, 7.1015, 7.1666, 7.3802, 6.901, 7.1927, 7.0963, 7.2396, 7.1823, 7.2214, 7.2682, 7.1693, 7.0573, 7.1328, 7.2057, 7.0911, 6.9714, 7.2344, 7.211, 7.237, 7.1276, 7.2422, 7.2317, 7.2474, 7.2031, 7.1536, 7.1172, 7.0182, 7.2135, 7.3229, 7.1198, 6.9713, 7.1823, 7.289, 7.2865, 7.1302, 7.1901, 7.125, 7.2162, 7.1927, 7.1432, 7.2135, 7.2031, 7.0703, 7.1771, 7.1302, 7.2083, 7.0208, 7.1745, 7.2708, 7.2734, 7.2135, 7.138, 7.1276, 7.1719, 7.2786, 7.138], "el_lr2e5": [7.1562, 7.0781, 6.9063, 7.1666, 6.9427, 6.9792, 7.0417, 6.7344, 7.2552, 7.1354, 7.0208, 7.2813, 7.1041, 7.1771, 7.1458, 7.2813, 7.0729, 7.1927, 6.9948, 7.125, 7.3229, 7.151, 7.1823, 7.2135, 7.0833, 7.0312, 7.2187, 7.1302, 7.2812, 7.2761, 7.1875, 7.0469, 7.1719, 7.3542, 7.1719, 7.026, 7.099, 7.1354, 7.1198, 7.3958, 7.2084, 7.0833, 7.4167, 7.2344, 7.1458, 7.25, 7.1875, 7.0938, 7.125, 7.3125, 7.2136, 7.2448, 7.125, 7.2292, 7.2448, 7.375, 7.1042, 7.125, 7.1094, 6.9896, 7.1719, 7.0938, 7.0469, 7.1562], "el_lr5e5": [7.1719, 7.2656, 7.0573, 7.3385, 7.1354, 6.9479, 7.0677, 6.8698, 7.2656, 7.2135, 7.1979, 7.2396, 7.026, 7.1458, 6.9896, 7.3646, 7.2969, 7.25, 7.1094, 7.3802, 7.1198, 7.4115, 7.2083, 7.2448, 7.0729, 6.9167, 7.1771, 7.1094, 6.9948, 7.0989, 7.2709, 7.0885, 7.0937, 7.4114, 7.1979, 7.0625, 7.1458, 7.2031, 7.2656, 7.2135, 7.2396, 7.1875, 7.3073, 7.2083, 7.25, 6.9792, 7.1667, 7.125, 7.1042, 7.1875, 7.1823, 7.375, 7.2083, 7.1927, 7.224, 7.2708, 7.1146, 7.1614, 7.0833, 6.9375, 7.1406, 7.2135, 7.0573, 7.1614], "el": [6.9757, 7.0035, 6.7865, 7.0816, 6.9444, 6.8733, 7.1545, 6.9253, 7.1059, 7.1806, 7.033, 7.1944, 6.967, 6.9809, 7.184, 7.0451, 7.1667, 7.1319, 7.2205, 7.1962, 7.2656, 7.1753, 7.2865, 7.2622, 7.2187, 7.0729, 7.0885, 7.1649, 7.1493, 7.1684, 7.1979, 7.3194, 7.0295, 7.2274, 7.2188, 7.1719, 7.0868, 7.2014, 7.191, 7.2309, 7.2066, 7.2396, 7.1458, 7.3264, 7.2378, 7.1476, 7.2135, 7.1684, 7.0139, 7.1389, 7.1441, 7.2187, 7.2726, 7.1719, 7.2309, 7.2778, 7.2257, 7.1701, 7.1892, 7.0938, 7.2552, 7.0851, 7.151, 7.1806], "full_critique": [7.0, 7.0547, 6.8542, 7.1146, 6.9453, 6.9531, 7.099, 6.8151, 7.1771, 7.1693, 7.0938, 7.151, 6.888, 7.0703, 7.2448, 7.1823, 6.987, 7.1719, 7.1953, 7.1537, 7.2266, 7.138, 7.2604, 7.336, 7.1224, 7.0365, 7.0156, 7.1016, 7.0156, 7.211, 7.2865, 7.2969, 7.1094, 7.2656, 7.3255, 7.2318, 7.099, 7.0365, 7.151, 7.2812, 7.1562, 7.1667, 7.3125, 7.2839, 7.2943, 7.1901, 7.289, 7.2396, 7.0312, 7.2448, 7.2031, 7.2031, 7.2526, 7.1927, 7.2214, 7.1563, 7.2474, 7.1068, 7.1562, 6.9948, 7.1901, 7.1406, 7.1667, 7.3099], "grpo_long": [7.0156, 7.0547, 6.7578, 7.1198, 6.9349, 6.8985, 7.1328, 6.8125, 7.0885, 7.1823, 6.9766, 7.1901, 6.8542, 7.0339, 7.1458, 7.086, 7.0417, 7.112, 7.1172, 7.099, 7.1615, 7.0911, 7.1901, 7.3542, 7.1328, 7.0677, 7.0755, 7.1146, 7.013, 7.2656, 7.2708, 7.3255, 7.1042, 7.2474, 7.2057, 7.2213, 7.2422, 7.1302, 7.2812, 7.1771, 7.1667, 7.2188, 7.2448, 7.276, 7.3854, 7.1276, 7.3359, 7.2266, 7.0625, 7.4193, 7.2969, 7.2943, 7.2995, 7.2292, 7.3802, 7.3307, 7.3255, 7.2214, 7.263, 7.0781, 7.2552, 7.1615, 7.3385, 7.3229, 7.401, 7.2188, 7.3541, 7.4271, 7.4323, 7.2865, 7.2188, 7.2188, 7.362, 7.3672, 7.401, 7.1146, 7.2943, 7.3802, 7.4349, 7.3385, 7.2604, 7.4245, 7.3385, 7.0547, 7.2396, 7.3985, 7.25, 7.2422, 7.3932, 7.3594, 7.4115, 7.2578, 7.3464, 7.3828, 7.2474, 7.2839, 7.2005, 7.2943, 7.2708, 7.237, 7.3125, 7.4427, 7.1302, 7.263, 7.4063, 7.4505, 7.2682, 7.3933, 7.3594, 7.3334, 7.3307, 7.4453, 7.401, 7.2526, 7.2839, 7.474, 7.2969, 7.4297, 7.3177, 7.3958, 7.4766, 7.4219, 7.5312, 7.2917, 7.4609, 7.4531, 7.4114, 7.474], "grpo_lr2e5": [7.1458, 7.2344, 6.8073, 7.1302, 6.849, 6.8958, 6.8594, 6.6562, 7.2552, 7.0938, 7.099, 7.2552, 7.0625, 7.1354, 6.8594, 7.4114, 7.0104, 7.3229, 7.125, 7.276, 7.2656, 7.2343, 7.3229, 7.3698, 7.0834, 6.9583, 7.3854, 7.0417, 7.1927, 7.2396, 7.2656, 7.1146, 7.2292, 7.2344, 7.1979, 6.9791, 7.1823, 7.1562, 7.25, 7.3958, 7.375, 7.276, 7.3333, 7.3906, 7.224, 7.2448, 7.3386, 7.3333, 7.3437, 7.5104, 7.4896, 7.3125, 7.2448, 7.3125, 7.3802, 7.3698, 7.3802, 7.2969, 7.3281, 7.2865, 7.3646, 7.2031, 7.276, 7.3437], "grpo_lr5e5": [7.0729, 7.0365, 6.9791, 7.224, 7.1719, 7.0312, 7.1302, 6.8802, 7.3229, 7.2396, 7.1667, 7.4167, 7.1875, 7.2656, 7.2448, 7.4792, 7.3021, 7.401, 7.1042, 7.3386, 7.3073, 7.3385, 7.4271, 7.3542, 7.2709, 7.2136, 7.3333, 7.1875, 7.2969, 7.3281, 7.4271, 7.1458, 7.2656, 7.4479, 7.3594, 7.2344, 7.375, 7.3646, 7.401, 7.5156, 7.4687, 7.4063, 7.4375, 7.5573, 7.4583, 7.3438, 7.4948, 7.3646, 7.2292, 7.5937, 7.5313, 7.6511, 7.4896, 7.5104, 7.4479, 7.4479, 7.5417, 7.4948, 7.4687, 7.4583, 7.5729, 7.4062, 7.4531, 7.5781], "grpo": [6.941, 7.0052, 6.8715, 7.033, 6.8854, 6.8333, 7.1232, 6.9305, 6.9045, 7.0972, 6.8871, 7.0937, 6.849, 6.9445, 7.0521, 7.0712, 7.0677, 7.0486, 7.1441, 7.1059, 7.1562, 7.0868, 7.1562, 7.2066, 7.2188, 7.0521, 7.0781, 7.1389, 7.092, 7.2153, 7.2448, 7.3073, 7.0608, 7.2327, 7.2118, 7.2569, 7.1632, 7.2257, 7.2396, 7.1615, 7.2465, 7.3594, 7.1736, 7.3698, 7.3056, 7.1997, 7.2552, 7.2639, 7.0868, 7.2691, 7.1788, 7.2361, 7.2379, 7.2674, 7.3663, 7.3264, 7.2327, 7.2101, 7.2274, 7.1979, 7.3281, 7.2135, 7.2743, 7.2917], "rubrics_only": [6.9844, 7.0417, 6.914, 7.0521, 6.9479, 6.9505, 7.2292, 6.9297, 7.1927, 7.2474, 7.1901, 7.2969, 7.0703, 7.0625, 7.2188, 7.2396, 7.0339, 7.1667, 7.1354, 7.2969, 7.237, 7.2005, 7.237, 7.3333, 7.1224, 7.1276, 7.1302, 7.1536, 7.0599, 7.1563, 7.276, 7.4193, 7.1901, 7.3515, 7.2135, 7.2396, 7.3073, 7.1901, 7.263, 7.2187, 7.1901, 7.263, 7.3177, 7.2474, 7.3594, 7.2344, 7.362, 7.2604, 7.0156, 7.125, 7.3151, 7.1901, 7.2812, 7.2552, 7.3229, 7.25, 7.2682, 7.2344, 7.2526, 7.0286, 7.2526, 7.0808, 7.2604, 7.2786]}
    ORDER_BIAS = {"heldout": {"order1_verdicts": {"A": 81, "B": 44, "TIE": 3}, "order2_verdicts": {"A": 91, "B": 34, "TIE": 3}}, "alpaca": {"order1_verdicts": {"A": 60, "B": 35, "TIE": 5}, "order2_verdicts": {"A": 54, "B": 42, "TIE": 4}}, "dolly": {"order1_verdicts": {"A": 42, "B": 19, "TIE": 3}, "order2_verdicts": {"A": 39, "B": 22, "TIE": 3}}}
    return MEAN_CURVES, ORDER_BIAS, RUNS


@app.cell
def _():
    import statistics as st

    import matplotlib.pyplot as plt
    import numpy as np

    COLORS = {"el": "#0072B2", "grpo": "#D55E00", "full_critique": "#009E73",
              "rubrics_only": "#CC79A7", "directive": "#E69F00"}
    LABELS = {"el": "EL (experience)", "grpo": "GRPO (scalar)",
              "full_critique": "full critique", "rubrics_only": "rubrics only",
              "directive": "directive (3.3-bit)"}
    SETS = ["heldout", "alpaca", "dolly"]
    SET_LABELS = {"heldout": "held-out in-domain", "alpaca": "OOD: AlpacaEval",
                  "dolly": "OOD: Dolly creative"}

    def group(runs, g):
        return [r for r in runs if r["group"] == g]

    def sem(v):
        return st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0

    return COLORS, LABELS, SETS, SET_LABELS, group, np, plt, sem, st


@app.cell
def _(mo):
    mo.md(r"""
    ## Headline: all rich-feedback arms beat the base model, but the scalar-reward arm wins

    Five arms share identical data, coach calls, LoRA config, optimizer-step counts, and LR.
    They differ only in **which part of the coach's feedback** trains the policy:

    | arm | learning signal | bandwidth |
    |---|---|---|
    | `grpo` | 1–10 scalar score, group-normalized REINFORCE | ~3.3 bits |
    | `el` | `<experience>` extraction conditions a frozen teacher; reverse-KL distillation | ~KB of text |
    | `full_critique` | entire critique as teacher context | ~KB of text |
    | `rubrics_only` | rubric text as teacher context | ~KB of text |
    | `directive` | 1-of-10 canned directive as teacher context | ~3.3 bits |
    """)
    return


@app.cell
def _(COLORS, LABELS, RUNS, SETS, SET_LABELS, group, np, plt, sem, st):
    def _headline():
        arms = ["el", "grpo", "full_critique", "rubrics_only", "directive"]
        fig, ax = plt.subplots(figsize=(9, 4.4))
        xs = np.arange(len(SETS))
        width = 0.15
        for i, arm in enumerate(arms):
            runs = [r for r in group(RUNS, arm) if r["steps"] <= 64]
            means = [st.mean([r[f"{s}_wr"] for r in runs]) for s in SETS]
            errs = [sem([r[f"{s}_wr"] for r in runs]) for s in SETS]
            pos = xs + (i - 2) * width
            ax.bar(pos, means, width * 0.88, color=COLORS[arm],
                   label=f"{LABELS[arm]} (n={len(runs)})")
            ax.errorbar(pos, means, yerr=errs, fmt="none", ecolor="#333", lw=1, capsize=2)
            for x, m in zip(pos, means):
                ax.text(x, m + 0.012, f"{m:.2f}".lstrip("0"), ha="center", fontsize=7)
        base = group(RUNS, "baseline")[0]
        ax.scatter(xs, [base[f"{s}_wr"] for s in SETS], marker="D", s=32,
                   facecolor="white", edgecolor="#555", zorder=4,
                   label="base vs base (noise floor)")
        ax.axhline(0.5, color="#555", lw=1, ls="--")
        ax.set_xticks(xs, [SET_LABELS[s] for s in SETS])
        ax.set_ylabel("win rate vs frozen base policy")
        ax.set_ylim(0.38, 0.72)
        ax.legend(fontsize=7.5, ncols=2)
        ax.set_title("64-step arms: win rate vs base under the held-out Qwen3-8B judge")
        fig.tight_layout()
        return fig
    _headline()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Read it: bars above the dashed 0.5 line beat the frozen base policy under the held-out
    judge; the white diamonds mark the base-vs-base noise floor. **GRPO (orange) leads on
    every set**; EL (blue) improves over base but sits significantly below GRPO
    (paired over 9 seeds: heldout t=4.4, AlpacaEval t=5.4, Dolly t=3.2). The 3.3-bit
    directive control (yellow) is the only arm at the noise floor in-domain — the one
    paper prediction that clearly survives at this scale.
    """)
    return


@app.cell
def _(COLORS, LABELS, RUNS, SETS, SET_LABELS, group, plt, sem, st):
    def _steps():
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
        for k, s in enumerate(SETS):
            ax = axes[k]
            for arm in ["el", "grpo"]:
                pts = {}
                for steps, g in [(64, arm), (128, arm + "_long")]:
                    runs = [r for r in group(RUNS, g)]
                    if g == arm:
                        runs = [r for r in runs if r["seed"] < 6 and r["steps"] <= 64]
                    if runs:
                        v = [r[f"{s}_wr"] for r in runs]
                        pts[steps] = (st.mean(v), sem(v))
                if len(pts) == 2:
                    x = sorted(pts)
                    ax.errorbar(x, [pts[i][0] for i in x], yerr=[pts[i][1] for i in x],
                                color=COLORS[arm], lw=2, marker="o", capsize=3,
                                label=LABELS[arm] if k == 0 else None)
            ax.axhline(0.5, color="#555", lw=1, ls="--")
            ax.set_xticks([64, 128])
            ax.set_xlabel("training steps")
            ax.set_title(SET_LABELS[s], fontsize=9)
            if k == 0:
                ax.set_ylabel("win rate vs base")
                ax.legend(fontsize=8)
        fig.suptitle("Doubling optimization: GRPO keeps improving; EL stays flat")
        fig.tight_layout()
        return fig
    _steps()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The divergence test (claim 2)

    The paper's mechanism predicts scalar-reward RL should over-optimize the training signal
    and *lose* generalization with continued training, while EL's high-bandwidth signal keeps
    transferring. Doubling training from 64 to 128 steps (2→4 epochs) shows the opposite here:
    **GRPO gains further on every set** (Dolly +5pp) while its training-score gain nearly
    doubles (+0.29 → +0.49); EL's numbers stay flat. The reward-hacking regime the paper
    describes was not reached at this scale — a real limitation of the downscale, but also a
    genuine failure to observe the claimed effect where the matched comparison was run.
    """)
    return


@app.cell
def _(COLORS, LABELS, MEAN_CURVES, np, plt):
    def _dyn():
        fig, ax = plt.subplots(figsize=(8, 3.8))
        W = 8
        for g, curve in MEAN_CURVES.items():
            base_arm = g.replace("_long", "")
            if base_arm not in COLORS or "lr" in g:
                continue
            ls = "--" if g.endswith("_long") else "-"
            sm = np.convolve(curve, np.ones(W) / W, mode="valid")
            label = LABELS[base_arm] + (" (128st)" if g.endswith("_long") else "")
            ax.plot(np.arange(len(sm)) + W / 2, sm, color=COLORS[base_arm], ls=ls,
                    lw=1.8, label=label)
        ax.set_xlabel("training step")
        ax.set_ylabel("coach score on training batches")
        ax.set_title("Training-signal dynamics (seed-mean, 8-step smoothing)")
        ax.legend(fontsize=7, ncols=2)
        fig.tight_layout()
        return fig
    _dyn()
    return


@app.cell
def _(ORDER_BIAS, mo):
    _rows = []
    for _s, _o in ORDER_BIAS.items():
        for _ord, _v in _o.items():
            _tot = _v["A"] + _v["B"] + _v["TIE"]
            _rows.append(f"| {_s} | {_ord.replace('_verdicts','')} | {_v['A']/_tot:.2f} | {_v['TIE']/_tot:.2f} |")
    mo.md(
        r"""
    ## Judge diagnostics: why order-swapping was mandatory

    Comparing the *identical* base policy against itself (different sampling seeds), the
    Qwen3-8B judge prefers whichever response sits in position A roughly 2:1 — in **both**
    orders. All reported win rates therefore average the two presentation orders, which
    cancels the bias (base-vs-base lands at 0.46–0.53, consistent with 0.5).

    | eval set | order | P(judge picks A) | P(tie) |
    |---|---|---|---|
    """ + "\n".join(_rows)
    )
    return


@app.cell
def _(RUNS, group, mo, sem, st):
    def _fmt(g, extra=""):
        runs = group(RUNS, g)
        if g in ("el", "grpo", "full_critique", "rubrics_only", "directive"):
            runs = [r for r in runs if r["steps"] <= 64]
        if not runs:
            return None
        cells = [f"`{g}`{extra}", str(len(runs))]
        for s in ["heldout", "alpaca", "dolly"]:
            v = [r[f"{s}_wr"] for r in runs]
            cells.append(f"{st.mean(v):.3f} ± {sem(v):.3f}")
        tg = [r["train_last8"] - r["train_first8"] for r in runs if r["train_last8"]]
        cells.append(f"{st.mean(tg):+.2f}" if tg else "—")
        return "| " + " | ".join(cells) + " |"

    _lines = [l for l in (
        _fmt("el"), _fmt("grpo"), _fmt("full_critique"), _fmt("rubrics_only"),
        _fmt("directive"), _fmt("el_long", " 128st"), _fmt("grpo_long", " 128st"),
        _fmt("el_lr2e5", " lr2e-5"), _fmt("el_lr5e5", " lr5e-5"),
        _fmt("grpo_lr2e5", " lr2e-5"), _fmt("grpo_lr5e5", " lr5e-5")) if l]
    mo.md(
        r"""
    ## Full results table

    Win rate vs base (mean ± s.e.m. over seeds), plus the coach-score gain on training
    batches (last 8 steps − first 8). Robustness rows: doubling steps and raising the LoRA
    learning rate do not close EL's gap to GRPO.

    | group | n | held-out | AlpacaEval | Dolly | trainΔ |
    |---|---|---|---|---|---|
    """ + "\n".join(_lines)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Verdict and limitations

    **Partially reproduced.** The low-bandwidth prediction holds (quantized directive ≈ no
    gain); the headline EL > scalar-RL advantage and the reward-hacking mechanism did not
    appear under this downscaled, matched setup. That reads as *"the effect did not show at
    this scale,"* not *"the paper is wrong"*:

    - **Scale**: 1.5B LoRA policy vs the paper's 7–8B full fine-tuning; 512 prompts vs 7,500;
      64–128 steps vs 3 epochs at batch 256×8.
    - **Teacher capacity**: the teacher is the frozen *initial 1.5B policy* conditioned on
      experience — a 1.5B model may extract far less value from textual guidance than an 8B
      one, directly capping EL's signal quality (the paper's own bandwidth argument cuts both
      ways: bandwidth only helps if the teacher can use it).
    - **Feedback model**: open Qwen2.5-7B coach instead of GPT-4o; judge is Qwen3-8B, not GPT-4o.
    - Reverse-KL/top-256 distillation details were re-implemented from the paper's description
      (no reference code).

    Compute: Kubernetes, 64 successful runs on NVIDIA RTX PRO 6000 Blackwell GPUs (1 GPU per
    run, peak 16 concurrent), ≈4.2 h wall clock end-to-end.
    """)
    return


if __name__ == "__main__":
    app.run()
