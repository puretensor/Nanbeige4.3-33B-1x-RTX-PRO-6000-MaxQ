#!/usr/bin/env bash
# Nanbeige4.3-33B on one RTX PRO 6000 Blackwell Max-Q: download the weights, start the server, smoke-test it.
#
#   ./start.sh            fast profile: NVFP4 + vLLM + n-gram drafts (94.1 tok/s code, 92.6 reasoning)
#   ./start.sh code       NVFP4 + SGLang + DFlash2 drafter (105.3 tok/s code, 71.5 reasoning)
#   ./start.sh --list     every profile with its measured numbers
#   ./start.sh code --print   print the exact docker command and exit
#
# Environment: GPU (default 0), PORT (8000), HOST (127.0.0.1), MODELS_DIR (./models), HF_TOKEN (optional).
set -euo pipefail
cd "$(dirname "$0")"

PROFILE="${1:-fast}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
MODELS_DIR="$(mkdir -p "${MODELS_DIR:-./models}" && cd "${MODELS_DIR:-./models}" && pwd)"
NAME="nanbeige-maxq"
SERVED="nanbeige43-33b"

# Images and weights exactly as measured (2026-11-02). Digests and commits are pinned so results reproduce.
VLLM_IMAGE="vllm/vllm-openai@sha256:66f19872e759e7d74f0729c4c7068eaaf71bc8b35f613573c36062a81be8cb3d"
SGLANG_IMAGE="lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469"
FP8="Nanbeige/Nanbeige4.3-33B-FP8@0b1acaa19bae10fec482600ada7c9defcb7c4884"
NVFP4="primitive-ai/Nanbeige4.3-33B-NVFP4@5ddf8788ed1fa4bec0c0b6fc74ddb0bef7dc4212"
DFLASH2="ProCreations/Nanbeige4.3-33B-DFlash2@22a73c4d099e61a34b6a26251b0a3a98a7b41746"

SGL_BASE="--context-length 16384 --mem-fraction-static 0.88 --max-running-requests 1 --chunked-prefill-size 2048 --attention-backend flashinfer --kv-cache-dtype bfloat16 --disable-radix-cache"

# profile | engine | weights | extra args | code-v1 | reasoning-v1
profiles() {
  cat <<'EOF'
fast          vllm    nvfp4  ngram      94.1   92.6
code          sglang  nvfp4  dflash6   105.3   71.5
fp8           vllm    fp8    ngram      60.0   58.1
fp8-dflash    sglang  fp8    dflash8    93.4   66.4
stock-nvfp4   vllm    nvfp4  -          67.7   67.7
stock-fp8     vllm    fp8    -          42.9   42.9
EOF
}

dir_of() { local r="${1%@*}"; echo "${r/\//__}"; }

fetch() {  # repo@commit -> $MODELS_DIR/<org>__<name>
  local repo="${1%@*}" rev="${1#*@}" dir; dir="$MODELS_DIR/$(dir_of "$1")"
  if [ -f "$dir/.pinned-$rev" ]; then return; fi
  local hf; hf="$(command -v hf || command -v huggingface-cli || true)"
  [ -n "$hf" ] || { echo "Need the Hugging Face CLI: pip install -U huggingface_hub" >&2; exit 1; }
  echo ">> downloading $repo at $rev into $dir"
  "$hf" download "$repo" --revision "$rev" --local-dir "$dir"
  touch "$dir/.pinned-$rev"
}

if [ "$PROFILE" = "--list" ] || [ "$PROFILE" = "-h" ] || [ "$PROFILE" = "--help" ]; then
  printf '%-13s %-7s %-7s %-9s %8s %13s\n' profile engine weights drafts code-v1 reasoning-v1
  profiles | awk '{printf "%-13s %-7s %-7s %-9s %8s %13s\n", $1,$2,$3,$4,$5,$6}'
  echo "tok/s, batch 1, one RTX PRO 6000 Blackwell Max-Q at 325 W. Sources in README.md."
  exit 0
fi

read -r _ ENGINE WEIGHTS DRAFTS CODE REASON < <(profiles | awk -v p="$PROFILE" '$1==p') || true
[ -n "${ENGINE:-}" ] || { echo "Unknown profile '$PROFILE'. Try ./start.sh --list" >&2; exit 1; }

case "$WEIGHTS" in nvfp4) MODEL="$NVFP4" ;; fp8) MODEL="$FP8" ;; esac
CMD=(docker run -d --name "$NAME" --gpus "\"device=$GPU\"" --ipc=host)
[ "$ENGINE" = sglang ] && CMD+=(--shm-size 32g)
CMD+=(-p "$HOST:$PORT:$([ "$ENGINE" = vllm ] && echo 8000 || echo 30000)"
      -v "$MODELS_DIR:/models:ro" -v "$MODELS_DIR/.cache-$ENGINE:/root/.cache")
[ -n "${HF_TOKEN:-}" ] && CMD+=(-e HF_TOKEN)
if [ "$ENGINE" = vllm ]; then
  CMD+=("$VLLM_IMAGE" --model "/models/$(dir_of "$MODEL")" --served-model-name "$SERVED" --trust-remote-code
        --max-model-len 32768 --gpu-memory-utilization 0.90)
  [ "$DRAFTS" = ngram ] && CMD+=(--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":1}'
                                 --attention-backend TRITON_ATTN)
else
  # shellcheck disable=SC2206  # SGL_BASE is a word list by design
  CMD+=("$SGLANG_IMAGE" python3 -m sglang.launch_server --model-path "/models/$(dir_of "$MODEL")" --served-model-name "$SERVED"
        --host 0.0.0.0 --port 30000 --trust-remote-code --enable-metrics $SGL_BASE)
  case "$DRAFTS" in
    dflash*) CMD+=(--speculative-algorithm DFLASH --speculative-draft-model-path "/models/$(dir_of "$DFLASH2")"
                   --speculative-draft-attention-backend flashinfer --speculative-draft-kv-cache-dtype bfloat16
                   --speculative-num-draft-tokens "${DRAFTS#dflash}") ;;
  esac
fi

if [ "${2:-}" = "--print" ]; then  # show the exact launch, change nothing
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

command -v docker >/dev/null || { echo "Docker is required." >&2; exit 1; }
grep -qi nvidia <<<"$(docker info 2>/dev/null)" || echo "!! Docker does not list an NVIDIA runtime; install the NVIDIA Container Toolkit if the start fails." >&2
FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
[ "$FREE" -ge 88000 ] || { echo "GPU $GPU has ${FREE} MiB free; these profiles want the whole 96 GB card (about 88 GB free)." >&2; exit 1; }

fetch "$MODEL"
case "$DRAFTS" in dflash*) fetch "$DFLASH2" ;; esac
docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "$MODELS_DIR/.cache-$ENGINE"
"${CMD[@]}" >/dev/null

echo ">> $PROFILE: $ENGINE, $WEIGHTS, drafts=$DRAFTS (measured $CODE tok/s code-v1, $REASON reasoning-v1)"
echo ">> waiting for http://$HOST:$PORT (first start compiles kernels and CUDA graphs: 3 to 10 minutes)"
for _ in $(seq 1 180); do
  curl -sf -m 5 "http://$HOST:$PORT/v1/models" >/dev/null && break
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || { docker logs --tail 60 "$NAME"; echo "!! server exited" >&2; exit 1; }
  sleep 10
done
curl -sf -m 5 "http://$HOST:$PORT/v1/models" >/dev/null || { docker logs --tail 60 "$NAME"; echo "!! not ready after 30 min" >&2; exit 1; }

REPLY=$(curl -s -m 120 "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word: ready\"}],\"max_tokens\":16,\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false,\"enable_thinking\":false}}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())')
echo ">> smoke test: $REPLY"
cat <<EOF
>> serving $SERVED at http://$HOST:$PORT/v1 (OpenAI-compatible)
   measure it:  python3 bench.py --url http://$HOST:$PORT --model $SERVED
   stop it:     ./stop.sh
EOF
