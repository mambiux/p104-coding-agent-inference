#!/bin/bash
# Qwen3.5-9B Q5_K_M (DeepSeek-V4-Pro distill) on the P104, upstream llama.cpp P104 build, 64k context. Mutually exclusive with bonsai-best (8 GB card).
export CUDA_VISIBLE_DEVICES=GPU-a2c2ef95-84b5-62ac-44b6-d862cdaf9b81
B=${B:-/home/mbx/CORE/llama/upstream-p61/bin}; M=${M:-/mnt/ssd500/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q5_K_M.gguf}; CTX=${CTX:-65536}; PORT=${PORT:-8098}
exec env LD_LIBRARY_PATH=$B $B/llama-server -m $M --host ${HOST:-0.0.0.0} --port $PORT -ngl 99 --ctx-size $CTX --parallel 1 -b 512 -ub 256 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --jinja --spec-type ngram-mod --temp 0.7 --top-p 0.95 --top-k 20 --cache-reuse 256 ${EXTRA:-}
