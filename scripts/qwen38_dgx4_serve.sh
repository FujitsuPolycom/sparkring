#!/usr/bin/env bash
# Start one rank of the Qwen3.8-27B EXL3 K5/K6 four-Spark candidate.
# Run inside the prepared runtime container on every rank. Rank 0 serves the
# API; ranks 1-3 run headless workers.
set -euo pipefail

rank=${RANK:?set RANK to 0, 1, 2, or 3}
rank0_rendezvous_addr=${RANK0_RENDEZVOUS_ADDR:?set RANK0_RENDEZVOUS_ADDR}
env_file=${QWEN_ENV_FILE:-/ws/rank.env}
model_path=${QWEN_MODEL_PATH:-/ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated}
chat_template=${QWEN_CHAT_TEMPLATE:-/ws/chat_template_agentic.jinja}
venv=${QWEN_VENV:-/ws/venv}
api_port=${QWEN_API_PORT:-8000}
expected_vllm_commit=229effc810ee6b8112f661472f6aace4eb8c787d
expected_exllamav3_commit=5f3c537ca9d89893d771256f5c43c93656553fbb
expected_clean_tree_sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
expected_exllamav3_diff_sha256=594b01547b0d801cf95926ea973719354150893121019aba2ad8832bc9f17fdb
expected_exllamav3_status_sha256=50d1a8375a38a705db327f9f17bcba5b269dfb55757958541c51b6c60193c4cc

case "$rank" in
    0|1|2|3) ;;
    *) echo "RANK must be 0, 1, 2, or 3; got $rank" >&2; exit 2 ;;
esac

test -f "$env_file" || {
    echo "rank environment file not found: $env_file" >&2
    exit 3
}
test -d "$model_path" || {
    echo "model directory not found: $model_path" >&2
    exit 4
}
test -f "$chat_template" || {
    echo "chat template not found: $chat_template" >&2
    exit 5
}
test -f "$venv/bin/activate" || {
    echo "Python environment not found: $venv" >&2
    exit 6
}
command -v pgrep >/dev/null 2>&1 || {
    echo "pgrep is required for the duplicate-engine guard; install procps" >&2
    exit 7
}

vllm_commit=$(git -c safe.directory=/ws/src/vllm-gg \
    -C /ws/src/vllm-gg rev-parse HEAD 2>/dev/null || true)
test "$vllm_commit" = "$expected_vllm_commit" || {
    echo "vLLM base commit mismatch: expected $expected_vllm_commit, got ${vllm_commit:-unavailable}" >&2
    exit 8
}
vllm_diff_sha256=$(git -c safe.directory=/ws/src/vllm-gg \
    -C /ws/src/vllm-gg diff --binary | sha256sum | awk '{print $1}')
vllm_status_sha256=$(git -c safe.directory=/ws/src/vllm-gg \
    -C /ws/src/vllm-gg status --porcelain --untracked-files=all | sha256sum | awk '{print $1}')
test "$vllm_diff_sha256" = "$expected_clean_tree_sha256" \
    && test "$vllm_status_sha256" = "$expected_clean_tree_sha256" || {
    echo "vLLM source tree differs from the pinned clean commit" >&2
    exit 9
}
exllamav3_commit=$(git -c safe.directory=/ws/src/exllamav3 \
    -C /ws/src/exllamav3 rev-parse HEAD 2>/dev/null || true)
test "$exllamav3_commit" = "$expected_exllamav3_commit" || {
    echo "ExLlamaV3 base commit mismatch: expected $expected_exllamav3_commit, got ${exllamav3_commit:-unavailable}" >&2
    exit 10
}
exllamav3_diff_sha256=$(git -c safe.directory=/ws/src/exllamav3 \
    -C /ws/src/exllamav3 diff --binary | sha256sum | awk '{print $1}')
exllamav3_status_sha256=$(git -c safe.directory=/ws/src/exllamav3 \
    -C /ws/src/exllamav3 status --porcelain --untracked-files=all | sha256sum | awk '{print $1}')
test "$exllamav3_diff_sha256" = "$expected_exllamav3_diff_sha256" \
    && test "$exllamav3_status_sha256" = "$expected_exllamav3_status_sha256" || {
    echo "ExLlamaV3 source tree does not match the pinned ARM patch" >&2
    exit 11
}

if pgrep -f '[v]llm serve' >/dev/null 2>&1; then
    echo "a vLLM serving process is already running in this container" >&2
    exit 12
fi

if [ "$rank" = 0 ]; then
    for port in "$api_port" 29500; do
        if ! "$venv/bin/python" -c \
            'import socket, sys; sock = socket.socket(); sock.bind(("0.0.0.0", int(sys.argv[1]))); sock.close()' \
            "$port" 2>/dev/null; then
            echo "required rank-0 port is already bound: $port" >&2
            exit 13
        fi
    done
fi

set -a
# shellcheck disable=SC1090
. "$env_file"
set +a
# shellcheck disable=SC1091
. "$venv/bin/activate"

endpoint_args=(--headless)
if [ "$rank" = 0 ]; then
    endpoint_args=(--host 0.0.0.0 --port "$api_port")
fi

exec "$venv/bin/vllm" serve "$model_path" \
    --served-model-name qwen38 \
    --quantization exl3 \
    --tensor-parallel-size 4 \
    --decode-context-parallel-size 1 \
    --nnodes 4 \
    --node-rank "$rank" \
    --master-addr "$rank0_rendezvous_addr" \
    --master-port 29500 \
    --distributed-executor-backend mp \
    --max-model-len 262144 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --async-scheduling \
    --scheduler-reserve-full-isl \
    --block-size 16 \
    --gpu-memory-utilization 0.70 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --attention-backend TRITON_ATTN \
    --mm-processor-kwargs '{"truncation":false}' \
    --mm-encoder-attn-backend TORCH_SDPA \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --chat-template "$chat_template" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    "${endpoint_args[@]}"
