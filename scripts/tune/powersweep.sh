#!/bin/bash
# Power-limit sweep on the P104 with the merged build + MTP head: decode and prefill tok/s, mean watts, tokens per joule. Runs as root (needs nvidia-smi -pl).
B=/home/<user>/CORE/llama/prism-pascal-best/bin; M=/home/<user>/models-local/Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp-v2.gguf; T=/home/<user>/longshot/tune
UUID=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | grep P104 | awk -F', ' '{print $2}')
while ! grep -q "KVQUALITY DONE" $T/kvquality.log; do sleep 15; done
pkill -f "llama-server.*--port 8097" 2>/dev/null; sleep 1
python3 -c "import json; print(json.dumps({'messages':[{'role':'user','content':'Write a complete, idiomatic Python module implementing an LRU cache class with get/put, TTL expiry, thread safety, type hints, docstrings and a small pytest suite. Output only code.'}],'max_tokens':600,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}))" > $T/gen600.json
python3 -c "import json; print(json.dumps({'messages':[{'role':'user','content':('The quick brown fox jumps over the lazy dog. '*400)+' Summarize in one sentence.'}],'max_tokens':8,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}))" > $T/pp4k.json
CUDA_VISIBLE_DEVICES=$UUID LD_LIBRARY_PATH=$B nohup $B/llama-server -m $M --host 127.0.0.1 --port 8097 -ngl 999 -c 16384 -np 1 -b 512 -ub 512 -fa on --jinja --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-p-min 0 --reasoning off --cache-reuse 0 > $T/power-server.log 2>&1 </dev/null &
for i in $(seq 1 120); do curl -s -m2 http://127.0.0.1:8097/health 2>/dev/null | grep -q ok && break; sleep 1; done
curl -s -m 300 http://127.0.0.1:8097/v1/chat/completions -H 'Content-Type: application/json' --data @$T/gen600.json > /dev/null   # warm-up
echo "limit_W | decode tok/s | mean W decode | tok/J decode | SM MHz | temp | prefill tok/s (4k prompt) | mean W prefill | tok/J prefill"
for pl in 180 165 150 135 120 110 105 90; do
  nvidia-smi -i 0 -pl $pl >/dev/null 2>&1; sleep 2
  nvidia-smi -i 0 --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits -lms 500 > $T/pw.csv 2>/dev/null & S=$!
  g=$(curl -s -m 600 http://127.0.0.1:8097/v1/chat/completions -H 'Content-Type: application/json' --data @$T/gen600.json | python3 /home/<user>/longshot/tune/pps.py); kill $S 2>/dev/null; sleep 0.3
  read wg cg tg <<< $(awk -F', ' 'NR>2{w+=$1;c+=$2;t=($3>t)?$3:t;n++} END{printf "%.0f %.0f %d", w/n, c/n, t}' $T/pw.csv)
  # prefill: fresh prompt each time (cache-reuse 0; vary the first word to defeat the prompt cache)
  sed "s/The quick brown fox/Prompt$pl quick brown fox/" $T/pp4k.json > $T/pp4k-$pl.json
  nvidia-smi -i 0 --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits -lms 500 > $T/pw2.csv 2>/dev/null & S=$!
  p=$(curl -s -m 600 http://127.0.0.1:8097/v1/chat/completions -H 'Content-Type: application/json' --data @$T/pp4k-$pl.json | python3 /home/<user>/longshot/tune/ppp.py); kill $S 2>/dev/null; sleep 0.3
  wp=$(awk -F', ' 'NR>2{w+=$1;n++} END{printf "%.0f", w/n}' $T/pw2.csv)
  echo "$pl | $g | $wg | $(python3 -c "print(round($g/$wg,3))") | $cg | $tg | $p | $wp | $(python3 -c "print(round($p/$wp,2))")"
done
nvidia-smi -i 0 -pl 180 >/dev/null 2>&1; pkill -f "llama-server.*--port 8097"; echo "POWER SWEEP DONE"
