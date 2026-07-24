import json
import random

import requests
from datasets import load_dataset

OPEN_ENDED_CATEGORIES = {"Generation", "Open QA", "Brainstorm"}
ALPACA_URL = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"


def _prompt_len_ok(tok, text, max_tokens):
    return len(tok(text, add_special_tokens=False)["input_ids"]) <= max_tokens


def load_prompt_sets(tok, cfg):
    """Returns dict of prompt lists. Split fixed with seed 0 regardless of run seed."""
    rng = random.Random(0)

    ds = load_dataset("HuggingFaceH4/no_robots")
    pool = []
    for split in ["train", "test"]:
        for ex in ds[split]:
            if ex.get("category") not in OPEN_ENDED_CATEGORIES:
                continue
            msgs = ex.get("messages") or []
            if not msgs or msgs[0].get("role") != "user":
                continue
            p = msgs[0]["content"].strip()
            if 20 <= len(p) and _prompt_len_ok(tok, p, cfg.MAX_PROMPT_TOKENS):
                pool.append(p)
    pool = sorted(set(pool))
    rng.shuffle(pool)
    need = cfg.N_TRAIN_PROMPTS + cfg.N_HELDOUT
    assert len(pool) >= need, f"no_robots pool too small: {len(pool)} < {need}"
    train = pool[: cfg.N_TRAIN_PROMPTS]
    heldout = pool[cfg.N_TRAIN_PROMPTS : need]

    try:
        alpaca_raw = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
        alpaca_all = [ex["instruction"] for ex in alpaca_raw]
    except Exception as e:
        print(f"alpaca via datasets failed ({e}); falling back to direct download", flush=True)
        alpaca_all = [ex["instruction"] for ex in json.loads(requests.get(ALPACA_URL, timeout=120).text)]
    alpaca_all = [p for p in alpaca_all if _prompt_len_ok(tok, p, cfg.MAX_PROMPT_TOKENS)]
    rng2 = random.Random(0)
    rng2.shuffle(alpaca_all)
    alpaca = alpaca_all[: cfg.N_ALPACA]

    dolly_ds = load_dataset("databricks/databricks-dolly-15k")["train"]
    dolly_pool = [
        ex["instruction"].strip()
        for ex in dolly_ds
        if ex["category"] == "creative_writing"
        and not ex.get("context")
        and len(ex["instruction"].strip()) >= 20
        and _prompt_len_ok(tok, ex["instruction"], cfg.MAX_PROMPT_TOKENS)
    ]
    dolly_pool = sorted(set(dolly_pool))
    rng3 = random.Random(0)
    rng3.shuffle(dolly_pool)
    dolly = dolly_pool[: cfg.N_DOLLY]

    sets = {"train": train, "heldout": heldout, "alpaca": alpaca, "dolly": dolly}
    print("DATA_SIZES " + json.dumps({k: len(v) for k, v in sets.items()}), flush=True)
    return sets
