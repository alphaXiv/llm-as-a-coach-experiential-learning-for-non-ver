import json
import random
import time

import torch

import config as cfg
from src import server as S
from src.data import load_prompt_sets
from src.evaluate import generate_eval_responses, judge_all
from src.policy import Policy
from src.train import train_arm


def main():
    t0 = time.time()
    print(f"GPU_INFO {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}", flush=True)
    print("CONFIG " + json.dumps({k: v for k, v in vars(cfg).items() if k.isupper()}), flush=True)
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    policy = Policy(cfg, with_lora=True)  # baseline also gets LoRA for smoke tests, then rebuilds
    data = load_prompt_sets(policy.tok, cfg)

    # ---- Phase 1: coach server (train-time feedback + rubric generation) ----
    coach = S.VLLMServer(cfg.COACH_MODEL, cfg.COACH_PORT, cfg.COACH_GPU_FRAC)
    coach.wait_ready()

    print("generating held-out rubrics...", flush=True)
    heldout_rubrics = S.gen_rubrics(coach, data["heldout"])

    history = []
    if cfg.ARM == "baseline":
        # smoke every arm's code path at tiny scale (results discarded)
        smoke_rubrics = S.gen_rubrics(coach, data["train"][:4])
        for arm in ["el", "grpo", "full_critique", "rubrics_only", "directive"]:
            print(f"SMOKE arm={arm}", flush=True)
            class C:
                pass
            c = C()
            for k in dir(cfg):
                if k.isupper():
                    setattr(c, k, getattr(cfg, k))
            c.ARM = arm
            train_arm(policy, coach, data["train"][:4], smoke_rubrics, c,
                      steps=2, prompts_per_step=2, samples_per_prompt=2,
                      max_new_tokens=128, tag="smoke")
        # rebuild without adapter so baseline eval is untouched base-vs-base
        del policy.model
        torch.cuda.empty_cache()
        policy = Policy(cfg, with_lora=False)
    else:
        print("generating train rubrics...", flush=True)
        train_rubrics = S.gen_rubrics(coach, data["train"])
        history = train_arm(policy, coach, data["train"], train_rubrics, cfg)

    # ---- Phase 2: eval generation with training model, then held-out judge ----
    eval_sets = {"heldout": data["heldout"], "alpaca": data["alpaca"], "dolly": data["dolly"]}
    responses = generate_eval_responses(policy, eval_sets, cfg)
    del policy.model
    torch.cuda.empty_cache()
    coach.stop()

    judge = S.VLLMServer(cfg.EVAL_JUDGE_MODEL, cfg.COACH_PORT + 1, cfg.JUDGE_GPU_FRAC, max_model_len=6144)
    judge.wait_ready()
    results = judge_all(judge, responses, heldout_rubrics, cfg)
    judge.stop()

    final = {
        "arm": cfg.ARM,
        "seed": cfg.SEED,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_hours": round((time.time() - t0) / 3600, 3),
        "train_history": history,
        "eval": results,
    }
    print("FINAL_RESULTS_JSON " + json.dumps(final), flush=True)
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
