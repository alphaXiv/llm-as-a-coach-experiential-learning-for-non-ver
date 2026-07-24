import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from src import prompts as P


class VLLMServer:
    def __init__(self, model, port, gpu_frac, max_model_len=4096):
        self.model = model
        self.port = port
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_frac),
            "--max-model-len", str(max_model_len),
            "--disable-log-requests",
        ]
        print(f"launching vllm: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout=1200):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm server for {self.model} exited early rc={self.proc.returncode}")
            try:
                r = requests.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
                if r.status_code == 200:
                    print(f"vllm ready ({self.model}) after {time.time()-t0:.0f}s", flush=True)
                    return
            except Exception:
                pass
            time.sleep(5)
        raise RuntimeError(f"vllm server for {self.model} not ready after {timeout}s")

    def stop(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=60)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        time.sleep(10)

    def chat(self, system, user, max_tokens=512, temperature=0.0, retries=3):
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "Qwen3" in self.model:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        for i in range(retries):
            try:
                r = requests.post(self.url, json=body, timeout=600)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if i == retries - 1:
                    print(f"chat call failed permanently: {e}", flush=True)
                    return ""
                time.sleep(5)

    def chat_many(self, items, max_tokens=512, temperature=0.0, workers=24):
        """items: list of (system, user). Preserves order."""
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(
                lambda su: self.chat(su[0], su[1], max_tokens=max_tokens, temperature=temperature),
                items,
            ))


def gen_rubrics(server, prompt_list):
    items = [(P.RUBRIC_SYSTEM, P.RUBRIC_USER.format(prompt=p)) for p in prompt_list]
    outs = server.chat_many(items, max_tokens=300, temperature=0.0)
    return [o.strip() if o.strip() else P.GENERIC_RUBRIC for o in outs]


FALLBACK_EXPERIENCE = ("Fully address every part of the request with a clear structure, "
                       "concrete specifics, and a tone matched to the task.")


def parse_coach(text):
    """Returns (score, analysis, experience, parse_ok)."""
    score = None
    m = re.search(r"[Ss]core:?\s*\**\s*(\d+(?:\.\d+)?)\s*/\s*10", text)
    if m:
        try:
            score = max(1.0, min(10.0, float(m.group(1))))
        except ValueError:
            score = None
    exp = None
    m2 = re.search(r"<experience>(.*?)</experience>", text, re.DOTALL)
    if m2 and m2.group(1).strip():
        exp = m2.group(1).strip()
    analysis = text.split("<experience>")[0].strip() if text else ""
    ok = score is not None and exp is not None
    return (score if score is not None else 5.0,
            analysis if analysis else FALLBACK_EXPERIENCE,
            exp if exp else FALLBACK_EXPERIENCE,
            ok)


def coach_feedback(server, triples, temperature=0.3):
    """triples: list of (prompt, rubric, response). Returns list of parse_coach outputs."""
    items = [(P.COACH_SYSTEM, P.COACH_USER.format(prompt=p, rubric=r, response=y))
             for p, r, y in triples]
    outs = server.chat_many(items, max_tokens=650, temperature=temperature)
    return [parse_coach(o) for o in outs]


def pick_directives(server, analyses):
    dir_list = "\n".join(f"{i+1}. {d}" for i, d in enumerate(P.DIRECTIVES))
    items = [(P.COACH_SYSTEM, P.DIRECTIVE_USER.format(analysis=a[:1500], directives=dir_list))
             for a in analyses]
    outs = server.chat_many(items, max_tokens=8, temperature=0.0)
    picks = []
    for o in outs:
        m = re.search(r"\d+", o or "")
        idx = int(m.group(0)) - 1 if m else 0
        idx = idx if 0 <= idx < len(P.DIRECTIVES) else 0
        picks.append(P.DIRECTIVES[idx])
    return picks
