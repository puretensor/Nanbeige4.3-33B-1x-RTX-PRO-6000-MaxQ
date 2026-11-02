#!/usr/bin/env python3
"""Batch-1 decode benchmark, measured the way LocalMaxxing verifies a run.

Fetches the canonical prompts (code-v1, reasoning-v1) from localmaxxing.com/api/agent-context and checks their
sha256, then streams each one at temperature 0, thinking off, 1,024 output tokens: 2 warmups discarded, median of 5.

  decode tok/s  = (completion_tokens - 1) / (t_last_token - t_first_token)
  prefill tok/s = prompt_tokens / time to first token

On vLLM it also reads the speculative-decoding counters from /metrics, so you get tokens per verify step
(the number that sets decode speed with drafts on: tok/s = tokens per step / seconds per step).

usage: python3 bench.py --url http://127.0.0.1:8000 --model nanbeige43-33b [--out results.jsonl]
"""
import argparse, hashlib, json, re, secrets, statistics, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="nanbeige43-33b")
ap.add_argument("--out", default="results.jsonl")
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--cases", default="code-v1,reasoning-v1")
a = ap.parse_args()

SPEC = re.compile(r"^(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens))(?:_total)?(?:\{[^}]*\})?\s+([0-9.e+]+)", re.M)


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "nanbeige-maxq-bench/1.0"})
    return urllib.request.urlopen(req, timeout=30).read().decode()


def spec_counters():
    try:
        txt = get(a.url + "/metrics")
    except Exception:
        return {}
    out = {}
    for name, val in SPEC.findall(txt):
        out[name] = out.get(name, 0.0) + float(val)
    return out


def run(prompt_text):
    # LocalMaxxing's cache-bust prefix: a fresh nonce per request so no prefix cache can help.
    nonce = f"[LocalMaxxing cache-bust nonce: {secrets.token_hex(16)}]\n"
    body = {"model": a.model, "messages": [{"role": "user", "content": nonce + prompt_text}],
            "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    before = spec_counters()
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = t_last = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    text.append(piece)
    after = spec_counters()
    spec = {k.split("num_")[1]: after.get(k, 0) - before.get(k, 0) for k in after} if after else {}
    ct, pt = usage["completion_tokens"], usage["prompt_tokens"]
    ttft = t_first - t0
    rec = {"prompt_tokens": pt, "completion_tokens": ct, "ttft_ms": ttft * 1000,
           "tok_s_out": (ct - 1) / (t_last - t_first), "tok_s_prefill": pt / ttft, "spec": spec,
           "output_sha256": hashlib.sha256("".join(text).encode()).hexdigest()}
    if spec.get("drafts"):
        # Tokens emitted per verify step, including the one the target adds itself.
        rec["tokens_per_step"] = 1 + spec["accepted_tokens"] / spec["drafts"]
        rec["ms_per_step"] = (t_last - t_first) * 1000 / spec["drafts"]
    return rec


ctx = json.loads(get("https://www.localmaxxing.com/api/agent-context"))
prompts = {}
for p in ctx["canonicalPrompts"]:
    assert hashlib.sha256(p["text"].encode()).hexdigest() == p["sha256"], f"{p['id']}: sha256 mismatch"
    prompts[p["id"]] = p["text"]

with open(a.out, "a") as f:
    for name in a.cases.split(","):
        for _ in range(a.warmup):
            run(prompts[name])
        recs = []
        for i in range(a.runs):
            r = run(prompts[name]); r.update(case=name, i=i)
            recs.append(r); f.write(json.dumps(r) + "\n"); f.flush()
        med = lambda k: statistics.median(x[k] for x in recs if x.get(k) is not None) if any(x.get(k) is not None for x in recs) else None
        line = (f"{name:13s} prompt={recs[0]['prompt_tokens']:4d} out={med('completion_tokens'):5.0f}  "
                f"decode={med('tok_s_out'):6.1f} tok/s  ttft={med('ttft_ms'):6.1f} ms  prefill={med('tok_s_prefill'):7.0f} tok/s")
        if med("tokens_per_step"):
            line += f"  {med('tokens_per_step'):.2f} tok/step at {med('ms_per_step'):.1f} ms/step"
        print(line, flush=True)
