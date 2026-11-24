#!/bin/bash
# decode speed vs context depth (q4_0 KV, merged build, with and without the MTP head), after the fill test
B=/home/<user>/CORE/llama/prism-pascal-best/bin; D=/home/<user>/models-local; T=/home/<user>/longshot/tune
UUID=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | grep P104 | awk -F', ' '{print $2}'); export CUDA_VISIBLE_DEVICES=$UUID LD_LIBRARY_PATH=$B
while ! grep -q "FILL TEST DONE" $T/filltest.log; do sleep 15; done
pkill -f "llama-server.*--port 809[7]" 2>/dev/null; sleep 2
echo "== tg64 vs depth, q4_0 KV, MTP file (bench = no speculative decoding), ub 256"
$B/llama-bench -m $D/Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp-v2.gguf -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ub 256 -p 0 -n 64 -d 0,8192,16384,24576,32768,49152,61440 -r 2 -o md 2>/dev/null | grep -E "tg64"
echo "== same with q8_0 KV (fits to 40k)"
$B/llama-bench -m $D/Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp-v2.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 256 -p 0 -n 64 -d 0,16384,32768 -r 2 -o md 2>/dev/null | grep -E "tg64"
echo "== f16 KV (fits to 20k)"
$B/llama-bench -m $D/Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp-v2.gguf -ngl 99 -fa 1 -ub 256 -p 0 -n 64 -d 0,16384 -r 2 -o md 2>/dev/null | grep -E "tg64"
echo "DEPTH SWEEP DONE"
