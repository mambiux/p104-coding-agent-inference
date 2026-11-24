#!/bin/bash
# fill-to-max stability: ctx 65536 q4_0 KV, prompt trimmed by /tokenize to ~60k tokens, generate 300. Runs after the power sweep.
B=/home/mbx/CORE/llama/prism-pascal-best/bin; M=/home/mbx/models-local/Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp-v2.gguf; T=/home/mbx/longshot/tune
UUID=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | grep P104 | awk -F', ' '{print $2}')
while ! grep -q "POWER SWEEP DONE" $T/powersweep.log; do sleep 15; done
pkill -f "llama-server.*--port 8097" 2>/dev/null; sleep 1
for cfg in "65536 q4_0" "49152 q4_0"; do set -- $cfg; CTX=$1; KV=$2
  CUDA_VISIBLE_DEVICES=$UUID LD_LIBRARY_PATH=$B nohup $B/llama-server -m $M --host 127.0.0.1 --port 8097 -ngl 999 -c $CTX -np 1 -b 512 -ub 256 -fa on --jinja -ctk $KV -ctv $KV --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-p-min 0 --reasoning off > $T/fill-server-$CTX.log 2>&1 </dev/null &
  for i in $(seq 1 120); do curl -s -m2 http://127.0.0.1:8097/health 2>/dev/null | grep -q ok && break; sleep 1; done
  python3 - $CTX $T <<'PY'
import json,sys,urllib.request,time
ctx=int(sys.argv[1]); T=sys.argv[2]; target=ctx-4500
txt=(open(f"{T}/corpus.txt").read()*4)
def ntok(s):
    r=urllib.request.Request("http://127.0.0.1:8097/tokenize",data=json.dumps({"content":s}).encode(),headers={"Content-Type":"application/json"})
    return len(json.load(urllib.request.urlopen(r,timeout=600))["tokens"])
lo,hi=1000,len(txt)
while hi-lo>2000:
    mid=(lo+hi)//2
    if ntok(txt[:mid])<target: lo=mid
    else: hi=mid
prompt=txt[:lo]; n=ntok(prompt); print(f"prompt trimmed to {n} tokens ({lo} chars)")
body={"messages":[{"role":"user","content":prompt+"\n\nIn one paragraph, what kinds of source files appear above and what do they implement?"}],"max_tokens":300,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
t0=time.time(); req=urllib.request.Request("http://127.0.0.1:8097/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
try:
    d=json.load(urllib.request.urlopen(req,timeout=3600)); t=d.get("timings",{}); c=d["choices"][0]["message"]["content"][:200].replace("\n"," ")
    print(f"FILL ctx={ctx}: prompt {t.get('prompt_n')} tok @ {t.get('prompt_per_second',0):.0f} tok/s ({t.get('prompt_ms',0)/1000:.0f}s), gen {t.get('predicted_n')} tok @ {t.get('predicted_per_second',0):.1f} tok/s, finish={d['choices'][0].get('finish_reason')}, wall {time.time()-t0:.0f}s | {c!r}")
    # second turn: append a follow-up to exercise cache reuse near the top
    body["messages"]+= [{"role":"assistant","content":d["choices"][0]["message"]["content"]},{"role":"user","content":"Now name three functions you saw and what each does."}]
    t0=time.time(); d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8097/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=3600)); t=d.get("timings",{})
    print(f"  follow-up turn: new prompt tokens {t.get('prompt_n')} @ {t.get('prompt_per_second',0):.0f} tok/s, gen {t.get('predicted_per_second',0):.1f} tok/s, wall {time.time()-t0:.0f}s | {d['choices'][0]['message']['content'][:120]!r}")
except Exception as e: print(f"FILL ctx={ctx} FAILED: {str(e)[:200]}")
PY
  echo "  peak VRAM: $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader) | server alive: $(pgrep -fc 'llama-server.*--port 8097') | server errors: $(grep -ciE 'out of memory|cudaMalloc failed|abort' $T/fill-server-$CTX.log)"
  pkill -f "llama-server.*--port 8097" 2>/dev/null; sleep 2
done
echo "FILL TEST DONE"
