import contextlib

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import prompts as P


class Policy:
    def __init__(self, cfg, with_lora=True):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.POLICY_MODEL)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.POLICY_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda()
        self.with_lora = with_lora
        if with_lora:
            lcfg = LoraConfig(
                r=cfg.LORA_R, lora_alpha=cfg.LORA_ALPHA, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lcfg)
            self.model.enable_input_require_grads()
        self.model.config.use_cache = True

    def base_ctx(self):
        """Context manager: run with adapters disabled (frozen initial policy)."""
        if self.with_lora:
            return self.model.disable_adapter()
        return contextlib.nullcontext()

    def student_prompt_ids(self, x):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": x}], add_generation_prompt=True)

    def teacher_prompt_ids(self, x, context):
        sys_msg = P.TEACHER_GUIDANCE_SYSTEM.format(context=context)
        return self.tok.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": x}],
            add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, user_prompts, max_new_tokens, temperature, use_base=False, batch_size=32):
        """Sample responses. Returns (list of response token id lists, list of texts)."""
        self.model.eval()
        all_ids, all_texts = [], []
        eos = self.tok.eos_token_id
        for i in range(0, len(user_prompts), batch_size):
            chunk = user_prompts[i : i + batch_size]
            id_lists = [self.student_prompt_ids(x) for x in chunk]
            maxlen = max(len(t) for t in id_lists)
            input_ids = torch.full((len(chunk), maxlen), self.tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for j, t in enumerate(id_lists):  # left pad
                input_ids[j, maxlen - len(t):] = torch.tensor(t)
                attn[j, maxlen - len(t):] = 1
            input_ids, attn = input_ids.cuda(), attn.cuda()
            ctx = self.base_ctx() if use_base else contextlib.nullcontext()
            with ctx:
                out = self.model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                    temperature=max(temperature, 1e-4), top_p=0.95,
                    pad_token_id=self.tok.pad_token_id,
                )
            new = out[:, maxlen:]
            for row in new.tolist():
                if eos in row:
                    row = row[: row.index(eos) + 1]
                row = [t for t in row if t != self.tok.pad_token_id or t == eos]
                all_ids.append(row)
                all_texts.append(self.tok.decode(row, skip_special_tokens=True).strip())
        return all_ids, all_texts

    def forward_micro(self, prompt_id_lists, y_id_lists, requires_grad, use_base=False):
        """Forward one micro-batch; returns per-sample logits sliced to response positions.

        Returns list of logits[T_y, V] (bf16, upcast at use site), None for empty responses.
        """
        seqs = [p + y for p, y in zip(prompt_id_lists, y_id_lists)]
        maxlen = max(len(t) for t in seqs)
        input_ids = torch.full((len(seqs), maxlen), self.tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for j, t in enumerate(seqs):  # right pad
            input_ids[j, : len(t)] = torch.tensor(t)
            attn[j, : len(t)] = 1
        input_ids, attn = input_ids.cuda(), attn.cuda()
        grad_ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()
        model_ctx = self.base_ctx() if use_base else contextlib.nullcontext()
        with grad_ctx, model_ctx:
            logits = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        out = []
        for j in range(len(seqs)):
            p_len, y_len = len(prompt_id_lists[j]), len(y_id_lists[j])
            out.append(None if y_len == 0 else logits[j, p_len - 1 : p_len + y_len - 1, :])
        return out
