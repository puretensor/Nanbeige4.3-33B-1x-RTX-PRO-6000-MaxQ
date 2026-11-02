# Nanbeige4.3-33B on one RTX PRO 6000 Blackwell Max-Q

**94.1 tok/s on code and 92.6 on reasoning, batch 1, on a 325 W Max-Q card.** Stock FP8 does 42.9.
With ProCreations' DFlash2 drafter, code reaches **105.3 tok/s**.

Every number below is a Verified run on [LocalMaxxing](https://www.localmaxxing.com), linked, so you can check the
output, the prompt and the exact launch command. `./start.sh` launches the same images, weights and flags.

**Recipe page, with the numbers explained:** https://puretensor.systems/models/Nanbeige4.3-33B-1x-RTX-PRO-6000-MaxQ/

```bash
git clone https://github.com/puretensor/Nanbeige4.3-33B-1x-RTX-PRO-6000-MaxQ
cd Nanbeige4.3-33B-1x-RTX-PRO-6000-MaxQ
./start.sh          # downloads 21 GB of NVFP4 weights, starts vLLM, smoke-tests it
python3 bench.py    # measures it the way LocalMaxxing does
```

## Performance

One RTX PRO 6000 Blackwell Max-Q (96 GB, 325 W), batch 1, temperature 0, thinking off, 1,024 output tokens,
median of 5 runs after 2 warmups.

| Profile | Weights | Engine | Drafts | code-v1 tok/s | reasoning-v1 tok/s | Confirmed gain over the same engine stock |
|---|---|---|---|---:|---:|---|
| `fast` (default) | NVFP4 | vLLM | n-gram, 3 tokens | [**94.1**](https://www.localmaxxing.com/runs/cmvyw8r1y099eoa01xq99kr01) | [**92.6**](https://www.localmaxxing.com/runs/cmvyw8rcg099hoa01t3x4khyy) | x1.37 [1.36-1.40] |
| `code` | NVFP4 | SGLang | DFlash2, 6 tokens | [**105.3**](https://www.localmaxxing.com/runs/cmvyw8kkx098woa01l92z4h01) | [71.5](https://www.localmaxxing.com/runs/cmvyw8kv3098zoa01zkr6ua68) | x1.75 [1.73-1.76] |
| `fp8-dflash` | FP8 | SGLang | DFlash2, 8 tokens | [93.4](https://www.localmaxxing.com/runs/cmvyw886e098noa01ny4j4g4y) | [66.4](https://www.localmaxxing.com/runs/cmvyw88r3098qoa01lr243sva) | x1.79 [1.75-1.82] |
| `stock-nvfp4` | NVFP4 | vLLM | none | [67.7](https://www.localmaxxing.com/runs/cmvyw8p990995oa017i6hxslk) | [67.7](https://www.localmaxxing.com/runs/cmvyw8phv0998oa01pss77jof) | baseline |
| `fp8` | FP8 | vLLM | n-gram, 3 tokens | [60.0](https://www.localmaxxing.com/runs/cmvywuzhd099woa01u456jtqe) | [58.1](https://www.localmaxxing.com/runs/cmvywuzro099zoa01fjtmu3gq) | x1.37 [1.36-1.39] |
| (not a profile) | NVFP4 | SGLang | none | [49.5](https://www.localmaxxing.com/runs/cmvyvhblc098doa01dw6wiu80) | [49.8](https://www.localmaxxing.com/runs/cmvyvhbuc098goa01hg73ofr1) | SGLang baseline |
| (not a profile) | FP8 | SGLang | none | [43.5](https://www.localmaxxing.com/runs/cmvyvgzak0981oa0188vetq9j) | [43.6](https://www.localmaxxing.com/runs/cmvyvgzxn0984oa013dizpil2) | SGLang baseline |
| `stock-fp8` | FP8 | vLLM | none | [42.9](https://www.localmaxxing.com/runs/cmvywuwjm099noa018kcrdkcc) | [42.9](https://www.localmaxxing.com/runs/cmvywuwsa099qoa01jggx09vw) | baseline |

- **Which to pick.** `fast` is the best all-rounder: drafts from the prompt itself cost nothing and help on any text.
  `code` is fastest on code but slower than `fast` on prose and reasoning.
- **Confirmed gain** is the ratio to the same engine's stock run, measured over interleaved fresh boots
  (stock, tuned, stock, tuned, stock, tuned). It is the geometric mean of the code-v1 and reasoning-v1 ratios, with a
  95% confidence interval from a hierarchical bootstrap (boots, then runs within each boot).
- **Long prompts.** With a 2,900-token prompt, `fast` still decodes at
  [92.2 tok/s](https://www.localmaxxing.com/runs/cmvyw8row099koa01n0nieprr) and `code` at
  [76.0](https://www.localmaxxing.com/runs/cmvyw8l450992oa01brmid6dw). LocalMaxxing marks those runs unverified only
  because that prompt is not one of its canonical ones.

## Why these numbers

**Stock is 42.9 because the card is reading weights.**
- Nanbeige4.3-33B is a dense 33B model, so every generated token reads every weight once.
- In FP8 that is 33.1 GB per token: the 35.5 GB checkpoint minus its 2.4 GB BF16 embedding table, of which only one row is read.
- At the card's 1.79 TB/s memory bandwidth, that caps decode near 54 tok/s. Stock reaches 79% of the cap.

**NVFP4 raises the cap.**
- [primitive-ai's NVFP4 checkpoint](https://huggingface.co/primitive-ai/Nanbeige4.3-33B-NVFP4) reads 19.7 GB per token, which moves the cap to about 91 tok/s.
- On vLLM it measures 67.7 with no drafts. The same checkpoint on SGLang 0.5.21 measures 49.5; we have not profiled why.

**N-gram drafts give tokens almost for free.**
- vLLM looks up the last few tokens in the prompt and the text so far, proposes up to 3 continuations, and the model
  checks them all in one forward pass.
- On code-v1, the 1,024 tokens took 760 verify steps: 1.35 tokens per step, at 14.3 ms per step.
- 1.35 / 0.0143 s = 94 tok/s. A step costs about the same as a plain decode step (1 / 67.7 = 14.8 ms), so the
  accepted drafts are close to free.
- Triton attention (`--attention-backend TRITON_ATTN`) is part of the confirmed config.

**DFlash2 is a real drafter, and it is strongest on code.**
- [ProCreations' DFlash2](https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2) is a 3-layer, 0.98B
  block-diffusion drafter. It proposes a whole block of tokens at once from the target's own hidden states.
- Code is predictable, so long blocks get accepted: 49.5 to 105.3 tok/s on code-v1. Prose and reasoning are less
  predictable: 49.8 to 71.5.
- It runs on stock SGLang 0.5.21 (`--speculative-algorithm DFLASH`).
- Our sweep tried 4, 6 and 8 draft tokens. 6 won on NVFP4, 8 on FP8.

**Why not Nanbeige4.2-3B as the drafter?** It looks like the obvious pick, and it does not work:
- Its embedding table is padded to 166,144 rows against the 33B's 166,912. The tokens themselves are identical, but
  vLLM refuses the pair.
- Its architecture is not native to vLLM either, so it would run on the slow Transformers fallback.

## Requirements

- Linux with Docker and the NVIDIA Container Toolkit.
- One RTX PRO 6000 Blackwell card (96 GB) with about 88 GB free. The numbers here are for the 325 W Max-Q.
- The `hf` CLI (`pip install -U huggingface_hub`).
- Disk space for the weights: 21 GB (NVFP4), 34 GB (FP8), plus 1.9 GB for DFlash2.
- Tested with NVIDIA driver 615.71.10, Docker 29.1 and NVIDIA Container Toolkit 1.20.1.

## Usage

```bash
./start.sh --list              # profiles and their measured numbers
./start.sh code                # start a profile
./start.sh fast --print        # print the exact docker command, start nothing
GPU=1 PORT=8001 ./start.sh     # pick the card and port
MODELS_DIR=/data/models ./start.sh
./stop.sh
```

The server is OpenAI-compatible at `http://127.0.0.1:8000/v1`, model name `nanbeige43-33b`. Bind it beyond localhost with `HOST=0.0.0.0`.

Pinned versions:

| What | Pin |
|---|---|
| vLLM image | `vllm/vllm-openai@sha256:66f19872e759...` (nightly `335dd77b0`) |
| SGLang image | `lmsysorg/sglang@sha256:b1259f3ea327...` (`v0.5.21-cu130`) |
| FP8 weights | [`Nanbeige/Nanbeige4.3-33B-FP8`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B-FP8) at `0b1acaa1` |
| NVFP4 weights | [`primitive-ai/Nanbeige4.3-33B-NVFP4`](https://huggingface.co/primitive-ai/Nanbeige4.3-33B-NVFP4) at `5ddf8788` |
| Drafter | [`ProCreations/Nanbeige4.3-33B-DFlash2`](https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2) at `22a73c4d` |

## How we measured

`bench.py` is the measurement, so you can compare your card with ours:
- **Prompts:** it fetches LocalMaxxing's canonical prompts and checks their hashes.
- **Settings:** batch 1, temperature 0, thinking off, 1,024 output tokens, 2 warmups, median of 5.
- **Decode:** (output tokens - 1) / (time from first to last token).
- **Prefill:** prompt tokens / time to first token. On these short prompts (about 270-330 tokens) it is bound by
  per-request setup, not compute, so it is not a useful prefill number.
- **On vLLM:** it also prints tokens per verify step and milliseconds per step, the two numbers that set decode speed when drafts are on.

**How we found the configs:**
- We used our tuning harness to sweep one setting at a time: drafts, attention backend, KV cache type, async scheduling.
- It gated each candidate on output quality and degeneration.
- It kept a change only if interleaved fresh-boot confirms showed a gain of at least 1%.
- Our CPU ran in a performance window (performance governor, boost on).

**Reproduced from a fresh clone** on the same card, with the CPU in its normal state (boost off), 2026-11-02:

| Profile | code-v1 | reasoning-v1 | tokens / step | Against the Verified run |
|---|---:|---:|---|---|
| `fast` | 95.2 | 92.8 | 1.36 at 14.3 ms | same, within 1.2% |
| `code` | 99.5 | 69.0 | n/a (SGLang) | 5.5% and 3.5% lower |

The vLLM profile runs as CUDA graphs and does not care about CPU boost. SGLang with DFlash2 does more work on the CPU
between steps, so expect it to lose a few percent on a host without boost.

## Credits

- [Nanbeige](https://huggingface.co/Nanbeige) for the model and its FP8 checkpoint.
- [primitive-ai](https://huggingface.co/primitive-ai) for the NVFP4 checkpoint.
- [ProCreations](https://huggingface.co/ProCreations) for DFlash2.
- [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang).
- [LocalMaxxing](https://www.localmaxxing.com) for the canonical prompts and verification.

Model weights are Apache-2.0 (see each model card) and are downloaded from Hugging Face, not redistributed here.
The scripts in this repository are MIT.

By [PureTensor](https://puretensor.ai). Results are posted on X as [@Dataphil3](https://x.com/Dataphil3).
