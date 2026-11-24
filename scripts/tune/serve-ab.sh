#!/bin/bash
# short-prompt serving A/B, interleaved 3x: ref build vs cand build vs cand with the FA kernel switched off. Isolates kernel effect on MTP acceptance and tok/s. ~10 min.
R=/home/<user>/CORE/llama/prism-pascal-best/bin; C=${1:-/home/<user>/CORE/llama/prism-pascal-best2b/bin}; T=/home/<user>/longshot/tune; M=/home/<user>/models-local/model-eng-dl/model-eng-Bonsai2-27B-Condensed-mtp-v2-thinkclose.gguf
srv(){ pkill -f "llama-server.*--port 8114" 2>/dev/null; sleep 1; env $2 LD_LIBRARY_PATH=$1 nohup $1/llama-server -m $M --host 127.0.0.1 --port 8114 -ngl 999 -c 49152 -np 1 -b 512 -ub 256 -fa on --jinja -ctk q8_0 -ctv q4_0 --spec-type ngram-mod,draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.75 --reasoning off --cache-reuse 256 > $T/ab-srv.log 2>&1 </dev/null & for i in $(seq 1 150); do curl -s -m2 http://127.0.0.1:8114/health 2>/dev/null | grep -q ok && return 0; sleep 1; done; return 1; }
gen(){ curl -s -m 600 http://127.0.0.1:8114/v1/chat/completions -H 'Content-Type: application/json' --data @$T/mv-short.json | python3 $T/parse_timings.py; }
for i in 1 2 3; do
  for cfg in "REF $R X=1" "CAND $C X=1" "CAND-noFA $C GGML_CUDA_FA_P61=0"; do set -- $cfg; lab=$1; b=$2; e=$3
    srv $b "$e" || { echo "$lab: server failed"; continue; }; gen >/dev/null; echo "$i $lab :: $(gen) :: $(nvidia-smi -i 0 --query-gpu=clocks.sm,temperature.gpu,memory.used --format=csv,noheader)"
  done
done
pkill -f "llama-server.*--port 8114"; echo "SERVE AB DONE"
