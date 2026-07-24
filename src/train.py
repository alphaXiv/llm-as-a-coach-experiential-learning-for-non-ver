import json
import random
import time

import torch

from src import server as S

CONTEXT_ARMS = {"el", "full_critique", "rubrics_only", "directive"}


def build_contexts(arm, coach_server, rubrics_b, feedback):
    """Per-sample teacher context string for context-distillation arms."""
    if arm == "el":
        return [fb[2] for fb in feedback]                     # <experience> extraction
    if arm == "full_critique":
        return [fb[1][:2500] for fb in feedback]              # full analysis text
    if arm == "rubrics_only":
        return list(rubrics_b)
    if arm == "directive":
        return S.pick_directives(coach_server, [fb[1] for fb in feedback])
    raise ValueError(arm)


def train_arm(policy, coach_server, train_prompts, train_rubrics, cfg,
              steps=None, prompts_per_step=None, samples_per_prompt=None,
              max_new_tokens=None, tag="train"):
    """One arm's full training loop. Matched across arms: same sampler, same
    optimizer step count, same LR, one optimizer step per training step."""
    arm = cfg.ARM if steps is None else "smoke_" + cfg.ARM
    arm_name = cfg.ARM
    steps = steps or cfg.TRAIN_STEPS
    B = prompts_per_step or cfg.PROMPTS_PER_STEP
    G = samples_per_prompt or cfg.SAMPLES_PER_PROMPT
    max_new = max_new_tokens or cfg.MAX_NEW_TOKENS

    params = [p for p in policy.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.LR, weight_decay=0.0)
    rng = random.Random(cfg.SEED)
    order = list(range(len(train_prompts)))
    rng.shuffle(order)
    cursor = 0
    t_start = time.time()
    history = []

    for step in range(steps):
        if time.time() - t_start > cfg.MAX_TRAIN_SECONDS:
            print(f"TRAIN_WALL_GUARD hit at step {step}; stopping early", flush=True)
            break
        # -- pick prompts (epoch-wise shuffled cycling) --
        idxs = []
        for _ in range(B):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            idxs.append(order[cursor])
            cursor += 1
        prompts_b = [train_prompts[i] for i in idxs for _ in range(G)]
        rubrics_b = [train_rubrics[i] for i in idxs for _ in range(G)]

        # -- on-policy sampling --
        torch.manual_seed(cfg.SEED * 100000 + step)
        y_ids, y_texts = policy.generate(prompts_b, max_new, cfg.TRAIN_TEMP, batch_size=32)
        keep = [i for i in range(len(y_ids)) if len(y_ids[i]) > 0 and y_texts[i]]
        if not keep:
            print(f"METRIC {json.dumps({'step': step, 'skipped': True})}", flush=True)
            continue

        # -- coach feedback (same call for every arm; arms differ in what they use) --
        feedback = S.coach_feedback(
            coach_server, [(prompts_b[i], rubrics_b[i], y_texts[i]) for i in keep])
        scores = [fb[0] for fb in feedback]
        parse_ok = sum(1 for fb in feedback if fb[3])

        policy.model.train()
        opt.zero_grad(set_to_none=True)
        loss_total, n_contrib = 0.0, 0

        if arm_name == "grpo":
            # scalar-reward GRPO: group-normalized advantages, token-mean REINFORCE
            adv = [0.0] * len(keep)
            pos = {k: j for j, k in enumerate(keep)}
            for g0 in range(0, len(prompts_b), G):
                group = [k for k in range(g0, g0 + G) if k in pos]
                if len(group) < 2:
                    continue
                rs = [scores[pos[k]] for k in group]
                mu = sum(rs) / len(rs)
                sd = (sum((r - mu) ** 2 for r in rs) / len(rs)) ** 0.5
                for k in group:
                    adv[pos[k]] = (scores[pos[k]] - mu) / (sd + 1e-4)
            sp_ids = [policy.student_prompt_ids(prompts_b[i]) for i in keep]
            sy_ids = [y_ids[i] for i in keep]
            n_valid = sum(1 for a in adv if a != 0.0) or 1
            for s in range(0, len(keep), cfg.MICRO_BATCH):
                sl = slice(s, s + cfg.MICRO_BATCH)
                if all(a == 0.0 for a in adv[sl]):
                    continue
                slices = policy.forward_micro(sp_ids[sl], sy_ids[sl], requires_grad=True)
                mb_loss = 0.0
                for j, lg in enumerate(slices):
                    a = adv[s + j]
                    if lg is None or a == 0.0:
                        continue
                    logp = torch.log_softmax(lg.float(), dim=-1)
                    tok_lp = logp.gather(1, torch.tensor(sy_ids[s + j], device=lg.device).unsqueeze(1)).squeeze(1)
                    mb_loss = mb_loss + (-a * tok_lp.mean()) / n_valid
                    n_contrib += 1
                if torch.is_tensor(mb_loss):
                    mb_loss.backward()
                    loss_total += mb_loss.item()
        else:
            # context distillation: reverse KL(student || teacher(context)) on top-K
            contexts = build_contexts(arm_name, coach_server, [rubrics_b[i] for i in keep], feedback)
            sp_ids = [policy.student_prompt_ids(prompts_b[i]) for i in keep]
            tp_ids = [policy.teacher_prompt_ids(prompts_b[i], contexts[j]) for j, i in enumerate(keep)]
            sy_ids = [y_ids[i] for i in keep]
            n_valid = len(keep)
            for s in range(0, len(keep), cfg.MICRO_BATCH):
                sl = slice(s, s + cfg.MICRO_BATCH)
                with torch.no_grad():
                    t_slices = policy.forward_micro(tp_ids[sl], sy_ids[sl], requires_grad=False, use_base=True)
                    t_logps = [None if lg is None else torch.log_softmax(lg.float(), dim=-1) for lg in t_slices]
                s_slices = policy.forward_micro(sp_ids[sl], sy_ids[sl], requires_grad=True)
                mb_loss = 0.0
                for j, lg in enumerate(s_slices):
                    if lg is None or t_logps[j] is None:
                        continue
                    s_logp = torch.log_softmax(lg.float(), dim=-1)
                    p_s = s_logp.exp()
                    topv, topi = p_s.topk(cfg.TOPK_KL, dim=-1)
                    s_top = s_logp.gather(1, topi)
                    t_top = t_logps[j].gather(1, topi)
                    kl_tok = (topv * (s_top - t_top)).sum(-1)
                    mb_loss = mb_loss + kl_tok.mean() / n_valid
                    n_contrib += 1
                if torch.is_tensor(mb_loss):
                    mb_loss.backward()
                    loss_total += mb_loss.item()
                del t_logps

        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP)
        opt.step()
        rec = {
            "tag": tag, "arm": arm, "step": step,
            "coach_score_mean": round(sum(scores) / len(scores), 4),
            "loss": round(loss_total, 6), "n_samples": len(keep),
            "n_contrib": n_contrib, "parse_ok": parse_ok,
            "grad_norm": round(float(gnorm), 4),
            "elapsed_s": round(time.time() - t_start, 1),
        }
        history.append(rec)
        print("METRIC " + json.dumps(rec), flush=True)

    return history
