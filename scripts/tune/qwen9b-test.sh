#!/bin/bash
# Qwen3.5-9B Q5_K_M on the P104 with upstream llama.cpp (P104 build): fit ladder, benches at depth, serving smoke. Pauses Bonsai serving, restores it at the end.
export CUDA_VISIBLE_DEVICES=GPU-a2c2ef95-84b5-62ac-44b6-d862cdaf9b81; ulimit -c 0
B=/home/mbx/CORE/llama/upstream-p61/bin; M=/mnt/ssd500/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q5_K_M.gguf; T=/home/mbx/longshot/tune
trap 'pkill -f "llama-server.*--port 8098" 2>/dev/null; sudo -n systemctl start bonsai-best; echo "[$(date +%T)] Bonsai serving restored"' EXIT
sudo -n systemctl stop bonsai-best; for i in $(seq 1 90); do [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 1; done; echo "card free: $(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader)"
echo "== model: $(LD_LIBRARY_PATH=$B $B/llama-bench -m $M -ngl 99 -p 0 -n 0 -r 1 -o md 2>&1 | grep -oE 'qwen[^|]*' | head -1)"
echo "== bench (KV q8_0/q8_0, ub 256): pp512 / tg128 at depth 0, 8192, 16384"
LD_LIBRARY_PATH=$B $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 256 -p 512 -n 128 -d 0,8192,16384 -r 2 -o md 2>/dev/null | grep -E "pp512|tg128" | sed 's/^| [^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|//' | tr -s ' '
echo "== bench (KV q4_0/q4_0): same"
LD_LIBRARY_PATH=$B $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ub 256 -p 512 -n 128 -d 0,16384 -r 2 -o md 2>/dev/null | grep -E "pp512|tg128" | sed 's/^| [^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|//' | tr -s ' '
echo "== context fit ladder (server load, KV q8/q8): 16k 24k 32k 40k"
for c in 16384 24576 32768 40960; do pkill -f "llama-server.*--port 8098" 2>/dev/null; sleep 1
  LD_LIBRARY_PATH=$B nohup $B/llama-server -m $M --host 127.0.0.1 --port 8098 -ngl 99 -c $c -np 1 -b 512 -ub 256 -fa on -ctk q8_0 -ctv q8_0 --jinja --spec-type ngram-mod > $T/qwen9b-srv.log 2>&1 </dev/null &
  ok=no; for i in $(seq 1 120); do curl -s -m2 http://127.0.0.1:8098/health 2>/dev/null | grep -q '"ok"' && { ok=yes; break; }; pgrep -f "llama-server.*--port 8098" >/dev/null || break; sleep 1; done
  echo "ctx $c: $ok | VRAM $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader) $(grep -oE 'out of memory' $T/qwen9b-srv.log | head -1)"; [ $ok = no ] && break; FIT=$c; done
echo "== serving smoke at ctx ${FIT:-none} (short code prompt, 400 tok, sampled)"
pkill -f "llama-server.*--port 8098" 2>/dev/null; sleep 1
LD_LIBRARY_PATH=$B nohup $B/llama-server -m $M --host 0.0.0.0 --port 8098 -ngl 99 -c ${FIT:-16384} -np 1 -b 512 -ub 256 -fa on -ctk q8_0 -ctv q8_0 --jinja --spec-type ngram-mod --temp 0.7 --top-p 0.95 --top-k 20 > $T/qwen9b-srv.log 2>&1 </dev/null &
for i in $(seq 1 120); do curl -s -m2 http://127.0.0.1:8098/health 2>/dev/null | grep -q '"ok"' && break; sleep 1; done
python3 - <<'PY' > $T/qwen9b-req.json
import json; json.dump({"messages":[{"role":"user","content":"Write a complete, idiomatic Python module implementing an LRU cache class with get/put, TTL expiry, thread safety, type hints, docstrings and a small pytest suite. Output only code."}],"max_tokens":400,"temperature":0.7},open('/dev/stdout','w'))
PY
curl -s -m 600 http://127.0.0.1:8098/v1/chat/completions -H 'Content-Type: application/json' --data @$T/qwen9b-req.json | python3 $T/parse_timings.py
echo "QWEN9B TEST DONE"
