#!/bin/bash
# context sweep on the merged build + MTP head. usage: ctxtest.sh  (logs to stdout)
B=/home/<user>/CORE/llama/prism-pascal-best/bin; M=/home/<user>/models-local/Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp-v2.gguf; T=/home/<user>/longshot/tune
UUID=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | grep P104 | awk -F', ' '{print $2}')
python3 -c "import json; print(json.dumps({'messages':[{'role':'user','content':('The quick brown fox jumps over the lazy dog. '*1200)+' Summarize the text above in one sentence.'}],'max_tokens':120,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}))" > $T/long.json
python3 -c "import json; print(json.dumps({'messages':[{'role':'user','content':'Write a Python function that parses ISO-8601 durations into seconds, with tests.'}],'max_tokens':300,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}))" > $T/short.json
try(){ pkill -f "llama-server.*--port 8097" 2>/dev/null; sleep 1
  CUDA_VISIBLE_DEVICES=$UUID LD_LIBRARY_PATH=$B nohup $B/llama-server -m $M --host 127.0.0.1 --port 8097 -ngl 999 -c $2 -np 1 -b 512 -ub $3 -fa on --jinja --cache-type-k $4 --cache-type-v $4 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-p-min 0 --reasoning off > /home/<user>/longshot/tune/ctxtest-server.log 2>&1 </dev/null &
  for i in $(seq 1 120); do curl -s -m2 http://127.0.0.1:8097/health 2>/dev/null | grep -q ok && break; pgrep -f "llama-server.*--port 8097" >/dev/null || break; sleep 1; done
  if ! curl -s -m2 http://127.0.0.1:8097/health 2>/dev/null | grep -q ok; then echo "$1 :: ctx=$2 ub=$3 kv=$4 :: FAILED ($(grep -oiE 'cudaMalloc failed|out of memory|allocating [0-9.]+ MiB[^:]*|not supported.{0,60}|error.{0,80}' /home/<user>/longshot/tune/ctxtest-server.log | head -1))"; return; fi
  v=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)
  g=$(curl -s -m 300 http://127.0.0.1:8097/v1/chat/completions -H 'Content-Type: application/json' --data @$T/short.json | python3 $T/parse_timings.py 2>&1 | tail -1)
  l=$(curl -s -m 900 http://127.0.0.1:8097/v1/chat/completions -H 'Content-Type: application/json' --data @$T/long.json | python3 $T/parse_long.py 2>&1 | tail -1)
  echo "$1 :: ctx=$2 ub=$3 kv=$4 :: VRAM $v after load :: short $g :: $l :: peak $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
}
try A-current   16384 512 f16
try B-ub256     20480 256 f16
try C-q8        32768 512 q8_0
try D-q8-ub256  40960 256 q8_0
try E-q4        49152 256 q4_0
try F-q4-max    65536 256 q4_0
try G-q8-max    49152 256 q8_0
pkill -f "llama-server.*--port 8097" 2>/dev/null; echo "CTX TESTS DONE"
