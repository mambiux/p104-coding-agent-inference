#!/bin/bash
# pause serving, run the device-plane restore proof + the -bs sampling A/B, resume serving
T=/home/mbx/longshot/tune; B=/home/mbx/CORE/llama/prism-pascal-best4/bin; M=/home/mbx/models-local/model-eng-dl/model-eng-Bonsai2-27B-Condensed-mtp-v2-thinkclose.gguf
sudo -n systemctl stop bonsai-best; sleep 2; for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do sudo -n kill -9 $p; done; sleep 1
echo "== $(date +%H:%M:%S) serving paused; free $(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader)"
echo "=== RESTORE PROOF"; bash /home/mbx/longshot/rs/restore-proof.sh 2>&1 | tail -60
grep -q "PROOF VOID" /home/mbx/longshot/rs/rp/*.log 2>/dev/null && { echo "=== retry with reasoning off"; EXTRA="--reasoning off" bash /home/mbx/longshot/rs/restore-proof.sh 2>&1 | tail -40; }
pkill -9 -f "llama-server.*--port 8113" 2>/dev/null; sleep 2
echo "=== BS A/B (sampler temp 0.7, serving recipe, 48k)"
srv(){ pkill -f "llama-server.*--port 8115" 2>/dev/null; sleep 1; HOST=127.0.0.1 PORT=8115 EXTRA="$1" nohup /home/mbx/longshot/bonsai-serve-best2.sh > $T/bs-srv.log 2>&1 </dev/null & for i in $(seq 1 150); do curl -s -m2 http://127.0.0.1:8115/health 2>/dev/null | grep -q '"ok"' && return 0; sleep 1; done; return 1; }
gen(){ curl -s -m 600 http://127.0.0.1:8115/v1/chat/completions -H 'Content-Type: application/json' --data @$1 | python3 $T/parse_timings.py; }
python3 - $T <<'PY'
import json,sys; T=sys.argv[1]
for n in ("mv-short","mv-long"):
    j=json.load(open(f"{T}/{n}.json")); j["temperature"]=0.7; j["top_p"]=0.95; j["top_k"]=20; j["seed"]=7; json.dump(j,open(f"{T}/{n}-sampled.json","w"))
PY
for rep in 1 2; do for cfg in "noBS " "BS -bs"; do set -- $cfg; lab=$1; e=${2:-}; srv "$e" || { echo "$lab server failed: $(tail -3 $T/bs-srv.log)"; continue; }; gen $T/mv-short-sampled.json >/dev/null; echo "$rep $lab short(sampled): $(gen $T/mv-short-sampled.json)"; [ $rep = 1 ] && echo "$rep $lab long(sampled):  $(gen $T/mv-long-sampled.json)"; done; done
pkill -f "llama-server.*--port 8115"; sleep 2
sudo -n systemctl start bonsai-best; for i in $(seq 1 240); do curl -s -m2 http://127.0.0.1:8097/health 2>/dev/null | grep -q '"ok"' && break; sleep 1; done
echo "== $(date +%H:%M:%S) serving resumed: $(systemctl is-active bonsai-best) after ${i}s, VRAM $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"; echo "TUNE48K DONE"
