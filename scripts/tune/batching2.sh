#!/bin/bash
# continuous-batching throughput per card: -np 1/2/4 with concurrent code-gen requests; serving config (q4_0 KV, ub 256, MTP depth 2, 128 W cap). CUDA_VISIBLE_DEVICES set by gpuq.
B=/home/mbx/CORE/llama/prism-pascal-best/bin; M=/home/mbx/models-local/Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp-v2.gguf; T=/home/mbx/longshot/tune
PROMPTS=("Write a Python module implementing an LRU cache with TTL expiry, thread safety, type hints and pytest tests. Output only code." "Write a TypeScript Express server with JWT auth, rate limiting, and tests with supertest. Output only code." "Write a Rust CLI that tails a log file and aggregates error counts per minute, with unit tests. Output only code." "Write a Go HTTP service exposing Prometheus metrics for a job queue, with table-driven tests. Output only code.")
for np in ${NPLIST:-1 2 4}; do
  ctx=$((49152 / np)); [ $np -eq 1 ] && ctx=49152
  pkill -f "llama-server.*--port 8108" 2>/dev/null; sleep 1
  LD_LIBRARY_PATH=$B nohup $B/llama-server -m $M --host 127.0.0.1 --port 8108 -ngl 999 -c $((ctx*np)) -np $np -b 512 -ub 256 -fa on --jinja -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-p-min 0 --reasoning off > $T/batch-server-$np.log 2>&1 </dev/null &
  for i in $(seq 1 150); do curl -s -m2 http://127.0.0.1:8108/health 2>/dev/null | grep -q ok && break; sleep 1; done
  curl -s -m2 http://127.0.0.1:8108/health | grep -q ok || { echo "np=$np: server failed: $(grep -iE 'out of memory|cudaMalloc' $T/batch-server-$np.log | head -1 | cut -c1-100)"; continue; }
  vram=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)
  nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits -lms 500 > $T/batch-pw.csv 2>/dev/null & S=$!
  t0=$(date +%s.%N); for j in $(seq 0 $((np-1))); do ( printf '%s' "${PROMPTS[$j]}" | python3 -c "import json,sys; print(json.dumps({'messages':[{'role':'user','content':sys.stdin.read()}],'max_tokens':300,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}))" > $T/batch-req-$j.json; curl -s -m 1200 http://127.0.0.1:8108/v1/chat/completions -H 'Content-Type: application/json' --data @$T/batch-req-$j.json > $T/batch-res-$j.json ) & done; wait; t1=$(date +%s.%N)
  kill $S 2>/dev/null; sleep 0.3; w=$(awk 'NR>1{s+=$1;n++} END{printf "%.0f", s/n}' $T/batch-pw.csv)
  tot=$(python3 -c "import json; print(sum(json.load(open('$T/batch-res-%d.json'%j))['usage']['completion_tokens'] for j in range($np)))"); per=$(python3 -c "import json; print(', '.join('%.1f'%json.load(open('$T/batch-res-%d.json'%j))['timings']['predicted_per_second'] for j in range($np)))")
  wall=$(python3 -c "print(round($t1-$t0,1))"); agg=$(python3 -c "print(round($tot/($t1-$t0),1))")
  echo "np=$np ctx/slot=$ctx :: VRAM $vram :: $tot tokens in ${wall}s = aggregate $agg tok/s (per-stream: $per) :: mean $w W :: $(python3 -c "print(round($agg/$w,3))") tok/J"
done
pkill -f "llama-server.*--port 8108" 2>/dev/null; echo "BATCHING DONE"
