#!/bin/bash
# KV-cache quantization quality: KLD of q8_0 / q4_0 KV vs f16 KV on a code corpus, ctx 8192, merged build. Then a fill-to-max stability run.
B=/home/mbx/CORE/llama/prism-pascal-best/bin; M=/home/mbx/models-local/Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp-v2.gguf; T=/home/mbx/longshot/tune
UUID=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | grep P104 | awk -F', ' '{print $2}'); export CUDA_VISIBLE_DEVICES=$UUID LD_LIBRARY_PATH=$B
while ! grep -q "CTX TESTS DONE" $T/ctxtest.log; do sleep 15; done
# corpus: real code + prose from the fork tree (~400 KB)
find /home/mbx/CORE/llama/prism-pascal-best/src/src /home/mbx/CORE/llama/prism-pascal-best/src/tools/server -name "*.cpp" -size +20k | head -12 | xargs cat > $T/corpus.txt; cat /home/mbx/CORE/llama/prism-pascal-best/src/README.md >> $T/corpus.txt; echo "corpus: $(wc -c < $T/corpus.txt) bytes"
echo "== base f16 KV"; $B/llama-perplexity -m $M -ngl 999 -fa on -c 8192 -b 512 --chunks 6 -f $T/corpus.txt --kl-divergence-base $T/kld_f16.bin 2>&1 | grep -E "Final estimate|estimate: PPL" | tail -1
for kv in q8_0 q4_0; do echo "== $kv KV vs f16"; $B/llama-perplexity -m $M -ngl 999 -fa on -c 8192 -b 512 --chunks 6 -f $T/corpus.txt -ctk $kv -ctv $kv --kl-divergence-base $T/kld_f16.bin --kl-divergence 2>&1 | grep -E "Mean +KLD|Maximum KLD|99.9%|Same top p|Mean +Δp|Mean PPL\(Q\)|PPL ratio|Final estimate" | head -8; done
# fill test: largest config that loaded in the sweep
if grep -q "^F-q4-max.*VRAM" $T/ctxtest.log; then CTX=65536; KV=q4_0; elif grep -q "^G-q8-max.*VRAM" $T/ctxtest.log; then CTX=49152; KV=q8_0; else CTX=49152; KV=q4_0; fi
echo "== FILL TEST ctx=$CTX kv=$KV: prompt ~$((CTX-4000)) tokens + 300 generated"
pkill -f "llama-server.*--port 8097" 2>/dev/null; sleep 1
nohup $B/llama-server -m $M --host 127.0.0.1 --port 8097 -ngl 999 -c $CTX -np 1 -b 512 -ub 256 -fa on --jinja -ctk $KV -ctv $KV --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-p-min 0 --reasoning off > $T/fill-server.log 2>&1 </dev/null &
for i in $(seq 1 120); do curl -s -m2 http://127.0.0.1:8097/health 2>/dev/null | grep -q ok && break; sleep 1; done
python3 - $CTX $T <<'PY'
import json,sys,urllib.request,time
ctx=int(sys.argv[1]); T=sys.argv[2]
words=int((ctx-4000)/1.3)  # ~1.3 tok per word for this corpus-ish filler
txt=(open(f"{T}/corpus.txt").read()*4)[:words*6]
body={"messages":[{"role":"user","content":txt+"\n\nIn one paragraph, what kinds of source files appear above and what do they implement?"}],"max_tokens":300,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
t0=time.time(); req=urllib.request.Request("http://127.0.0.1:8097/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
try:
    d=json.load(urllib.request.urlopen(req,timeout=3000)); t=d.get("timings",{}); c=d["choices"][0]["message"]["content"][:160].replace("\n"," ")
    print(f"fill: prompt {t.get('prompt_n')} tok @ {t.get('prompt_per_second',0):.0f} tok/s ({t.get('prompt_ms',0)/1000:.0f}s), gen {t.get('predicted_n')} tok @ {t.get('predicted_per_second',0):.1f} tok/s, finish={d['choices'][0].get('finish_reason')} | {c!r}")
except Exception as e: print("fill FAILED:",str(e)[:200])
PY
echo "peak VRAM: $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader) | server alive: $(pgrep -fc 'llama-server.*--port 8097') | errors: $(grep -ciE 'out of memory|cudaMalloc failed|abort' $T/fill-server.log)"
pkill -f "llama-server.*--port 8097"; echo "KVQUALITY DONE"
