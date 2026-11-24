#!/bin/bash
# Qwen3.5-9B Q5_K_M at 64k context on the P104: load, 30k-token turn, short turn, VRAM headroom; depth benches at 32k/49k. Restores Bonsai at the end.
export CUDA_VISIBLE_DEVICES=GPU-a2c2ef95-84b5-62ac-44b6-d862cdaf9b81; ulimit -c 0
B=/home/mbx/CORE/llama/upstream-p61/bin; M=/mnt/ssd500/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q5_K_M.gguf; T=/home/mbx/longshot/tune
trap 'pkill -f "llama-server.*--port 8098" 2>/dev/null; sudo -n systemctl start bonsai-best; echo "[$(date +%T)] Bonsai serving restored"' EXIT
sudo -n systemctl stop bonsai-best; for i in $(seq 1 90); do [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 1; done
echo "== decode at depth 32k / 49k (KV q8/q8)"
LD_LIBRARY_PATH=$B $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 256 -p 512 -n 64 -d 32768,49152 -r 1 -o md 2>/dev/null | grep -E "pp512|tg64" | sed 's/^| [^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|//' | tr -s ' '
echo "== serve at 65536 (KV q8/q8, ub 256, ngram-mod)"
LD_LIBRARY_PATH=$B nohup $B/llama-server -m $M --host 0.0.0.0 --port 8098 -ngl 99 -c 65536 -np 1 -b 512 -ub 256 -fa on -ctk q8_0 -ctv q8_0 --jinja --spec-type ngram-mod --temp 0.7 --top-p 0.95 --top-k 20 > $T/qwen9b-srv.log 2>&1 </dev/null &
for i in $(seq 1 150); do curl -s -m2 http://127.0.0.1:8098/health 2>/dev/null | grep -q '"ok"' && break; pgrep -f "llama-server.*--port 8098" >/dev/null || break; sleep 1; done
echo "loaded: $(curl -s -m3 http://127.0.0.1:8098/health) | idle VRAM $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader) | n_ctx $(curl -s -m5 http://127.0.0.1:8098/props | python3 -c "import json,sys; print(json.load(sys.stdin)['default_generation_settings']['n_ctx'])" 2>/dev/null)"
echo "30k turn: $(curl -s -m 900 http://127.0.0.1:8098/v1/chat/completions -H 'Content-Type: application/json' --data @$T/mv-long.json | python3 $T/parse_timings.py) | VRAM $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
echo "short:    $(curl -s -m 600 http://127.0.0.1:8098/v1/chat/completions -H 'Content-Type: application/json' --data @$T/mv-short.json | python3 $T/parse_timings.py) | VRAM $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader) | restarts/crash: $(grep -cE 'out of memory|CUDA error' $T/qwen9b-srv.log)"
echo "QWEN9B 64K DONE"
