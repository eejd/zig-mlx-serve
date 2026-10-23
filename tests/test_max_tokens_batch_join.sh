#!/bin/bash
# max_tokens stays a hard cap when another request's prefill completes on a stream's last token (#16).
#
# Regression for: a stream that emitted its last budgeted token in a serial tick owed one more
# (finishing) tick; when a second request joined `decoding` in between, the finishing tick was a
# batched one, which forwarded the stream (no pipeline state to drain) and emitted max_tokens+1
# ("<- 1058+257 tokens streamed ... [length]", guidellm output_tokens 257 of 256).
#
# The window is one decode tick wide, so tests/test_max_tokens_batch_join.py sweeps the arrival time
# of the second request across the first one's lifetime, with and without ignore_eos.
#
# Requires: a built mlx-serve (zig build -Doptimize=ReleaseFast) and a small chat model:
#   MAXTOK_TEST_MODEL (default ~/.mlx-serve/models/mlx-community/Llama-3.2-1B-Instruct-4bit)
#
# Usage: ./tests/test_max_tokens_batch_join.sh [port]

set -e

PORT=${1:-8098}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${MAXTOK_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/Llama-3.2-1B-Instruct-4bit}"
if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_max_tokens_batch_join: model directory not found."
    echo "  Set MAXTOK_TEST_MODEL or place a checkpoint at ~/.mlx-serve/models/mlx-community/Llama-3.2-1B-Instruct-4bit."
    exit 0
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

LOGFILE=$(mktemp)
"$BINARY" --model "$MODEL" --serve --port "$PORT" --host 127.0.0.1 > "$LOGFILE" 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null || true; }
trap cleanup EXIT

up=0
for i in $(seq 1 60); do
    if curl -s -f "$BASE/health" > /dev/null 2>&1; then up=1; break; fi
    sleep 1
done
if [ "$up" != "1" ]; then
    echo -e "${RED}FAIL${NC} server did not become healthy in 60s"
    tail -20 "$LOGFILE"
    exit 1
fi

if python3 "$(dirname "$0")/test_max_tokens_batch_join.py" "$PORT"; then
    rm -f "$LOGFILE"
    echo -e "${GREEN}PASS${NC} every stream ended at or below its max_tokens"
else
    echo -e "${RED}FAIL${NC} a stream passed its max_tokens (server log: $LOGFILE)"
    exit 1
fi
