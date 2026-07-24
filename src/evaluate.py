import json
import re

import torch

from src import prompts as P


def parse_verdict(text):
    t = (text or "").strip().upper()
    m = re.search(r"\b(A|B|TIE)\b", t)
    return m.group(1) if m else "TIE"


def parse_score(text):
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text or "")
    if not m:
        m = re.search(r"\b(\d+(?:\.\d+)?)\b", text or "")
    if not m:
        return None
    try:
        return max(1.0, min(10.0, float(m.group(1))))
    except ValueError:
        return None


def generate_eval_responses(policy, eval_sets, cfg):
    """Policy (adapter on) and base (adapter off) samples for every eval prompt."""
    out = {}
    for name, prompts in eval_sets.items():
        torch.manual_seed(cfg.SEED * 7919 + 1)
        _, pol = policy.generate(prompts, cfg.EVAL_MAX_NEW_TOKENS, cfg.EVAL_TEMP, batch_size=32)
        torch.manual_seed(cfg.SEED * 7919 + 1000)
        _, base = policy.generate(prompts, cfg.EVAL_MAX_NEW_TOKENS, cfg.EVAL_TEMP,
                                  use_base=True, batch_size=32)
        out[name] = {"prompts": prompts, "policy": pol, "base": base}
        print(f"generated eval responses for {name}: {len(prompts)}", flush=True)
    return out


def judge_all(judge, responses, heldout_rubrics, cfg):
    """Order-swapped pairwise policy-vs-base + absolute rubric scores."""
    results = {}
    for name, d in responses.items():
        prompts, pol, base = d["prompts"], d["policy"], d["base"]
        n = len(prompts)
        # pairwise, both orders (order1: A=policy, order2: A=base)
        items = []
        for i in range(n):
            items.append((P.PAIRWISE_SYSTEM, P.PAIRWISE_USER.format(prompt=prompts[i], a=pol[i], b=base[i])))
        for i in range(n):
            items.append((P.PAIRWISE_SYSTEM, P.PAIRWISE_USER.format(prompt=prompts[i], a=base[i], b=pol[i])))
        outs = judge.chat_many(items, max_tokens=8, temperature=0.0)
        v1 = [parse_verdict(o) for o in outs[:n]]
        v2 = [parse_verdict(o) for o in outs[n:]]
        wins = losses = ties = 0
        per_prompt = []
        for i in range(n):
            s = 0.0
            s += {"A": 1.0, "TIE": 0.5, "B": 0.0}[v1[i]]
            s += {"B": 1.0, "TIE": 0.5, "A": 0.0}[v2[i]]
            s /= 2.0
            per_prompt.append(s)
            wins += s == 1.0
            losses += s == 0.0
            ties += 0.0 < s < 1.0
        win_rate = sum(per_prompt) / n if n else 0.0

        # absolute scores (per-prompt rubrics in-domain, generic rubric OOD)
        rubs = heldout_rubrics if name == "heldout" else [P.GENERIC_RUBRIC] * n
        sc_items = [(P.SCORE_SYSTEM, P.SCORE_USER.format(prompt=prompts[i], rubric=rubs[i], response=pol[i]))
                    for i in range(n)]
        sc_items += [(P.SCORE_SYSTEM, P.SCORE_USER.format(prompt=prompts[i], rubric=rubs[i], response=base[i]))
                     for i in range(n)]
        sc_outs = judge.chat_many(sc_items, max_tokens=16, temperature=0.0)
        pol_scores = [parse_score(o) for o in sc_outs[:n]]
        base_scores = [parse_score(o) for o in sc_outs[n:]]
        pol_valid = [s for s in pol_scores if s is not None]
        base_valid = [s for s in base_scores if s is not None]

        results[name] = {
            "n": n,
            "win_rate": round(win_rate, 4),
            "wins": int(wins), "losses": int(losses), "ties": int(ties),
            "order1_verdicts": {v: v1.count(v) for v in ["A", "B", "TIE"]},
            "order2_verdicts": {v: v2.count(v) for v in ["A", "B", "TIE"]},
            "per_prompt_scores": per_prompt,
            "policy_abs_score": round(sum(pol_valid) / max(len(pol_valid), 1), 4),
            "base_abs_score": round(sum(base_valid) / max(len(base_valid), 1), 4),
            "score_parse_fail": (len(pol_scores) - len(pol_valid)) + (len(base_scores) - len(base_valid)),
        }
        print("EVAL_SET " + json.dumps({k: v for k, v in results[name].items() if k != "per_prompt_scores"} | {"set": name}), flush=True)
    return results
