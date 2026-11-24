#!/bin/bash
# Bonsai 27B on the P104 — program-2 serving recipe (2026-11-24). Needs the merged build prism-pascal-best2 (FA p61 kernel + mixed-KV FA instance).
# Decisions: thinkclose serving file; K=q8_0/V=q4_0 KV (KLD 4e-4 vs f16, ~68k fits); 64k ctx; n-gram + MTP depth 3, p-min 0.75; reasoning medium + 4k budget with wrap-up message; 128 W cap (systemd unit).
B=${B:-/home/<user>/CORE/llama/prism-pascal-best2/bin}; UUID=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | grep P104 | awk -F', ' '{print $2}')
MODEL=${MODEL:-/home/<user>/models-local/model-eng-dl/model-eng-Bonsai2-27B-Condensed-mtp-v2-thinkclose.gguf}; CTX=${CTX:-49152}; PORT=${PORT:-8097}; KVK=${KVK:-q4_0}; KVV=${KVV:-q4_0}; UB=${UB:-256}
exec env CUDA_VISIBLE_DEVICES=$UUID LD_LIBRARY_PATH=$B $B/llama-server -m $MODEL --host ${HOST:-0.0.0.0} --port $PORT -ngl 999 --ctx-size $CTX --parallel 1 -b 512 -ub $UB -fa on --cache-type-k $KVK --cache-type-v $KVV --jinja \
  --spec-type ngram-mod,draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.75 \
  --reasoning on --reasoning-effort medium --reasoning-budget 4000 --reasoning-budget-message "Time to stop thinking and write the final answer now." \
  --temp 0.7 --top-p 0.95 --top-k 20 --cache-reuse 256 ${EXTRA:-}
