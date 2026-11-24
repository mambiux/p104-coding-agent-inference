#!/bin/bash
# Conservative memory-offset ladder for the P104: +50..+200 MHz, hard gates, always rolls back. Serving is stopped for the run and restarted after.
N=/home/mbx/CORE/llama/prism-pascal-novel; B=/home/mbx/CORE/llama/prism-pascal-best8/bin; T=/home/mbx/longshot/tune
M=/home/mbx/models-local/model-eng-dl/model-eng-Bonsai2-27B-Condensed-mtp-v2-thinkclose.gguf
set -o pipefail
export CUDA_VISIBLE_DEVICES=GPU-a2c2ef95-84b5-62ac-44b6-d862cdaf9b81   # the P104 — CUDA enumerates the 3060 first, nvidia-smi -i 0 is the P104
ulimit -c 0
rollback(){ python3 -I $N/scripts/memoc_set.py 0 | tail -1; systemctl start bonsai-best; echo "[$(date +%T)] rolled back to stock, serving restarted"; }
trap rollback EXIT
systemctl stop bonsai-best; for i in $(seq 1 90); do [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 1; done
echo "serving stopped after ${i}s; P104 free: $(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader) | unit: $(systemctl is-active bonsai-best)"
xid0=$(dmesg | grep -c Xid)
gate(){ # $1 = offset label
  t=$(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$t" -gt 80 ] && { echo "GATE: temp ${t}C > 80 at $1"; return 1; }
  [ "$(dmesg | grep -c Xid)" -gt "$xid0" ] && { echo "GATE: new Xid in dmesg at $1"; dmesg | grep Xid | tail -2; return 1; }
  return 0
}
step(){ o=$1; echo "=== offset +$o  $(date +%T)"
  python3 -I $N/scripts/memoc_set.py $o | tr '\n' ' '; echo
  echo "clocks: $(nvidia-smi -i 0 --query-gpu=clocks.mem,clocks.sm,power.draw,temperature.gpu --format=csv,noheader)"
  mt=$($N/out/memtest 8 2>&1 | tail -1); echo "$mt"; echo "$mt" | grep -q "bit errors = 0," || { echo "GATE: MEMTEST ERRORS at +$o"; return 1; }
  $N/out/hwprobe 2>/dev/null | grep -E "DRAM read grid 240" | head -1
  LD_LIBRARY_PATH=$B $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ub 256 -p 512 -n 128 -r 3 -o md 2>/dev/null | grep -E "pp512|tg128" | sed 's/^| [^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|//' | tr -s ' ' | tr '\n' ' '; echo
  LD_LIBRARY_PATH=$B $B/llama-perplexity -m $M -ngl 999 -fa on -c 512 -b 512 -ub 1 --chunks 4 -f $T/corpus.txt -ctk q4_0 -ctv q4_0 --kl-divergence-base $T/kld_memoc_base.bin --kl-divergence > $T/memoc-ppl.out 2>&1; k=$(grep -E "Mean    KLD|Same top p" $T/memoc-ppl.out | tr -s ' ' | tr '\n' ' '); echo "$k"; [ -z "$k" ] && { echo "perplexity produced no KLD line; tail:"; tail -3 $T/memoc-ppl.out | cut -c1-160; }
  kld=$(echo "$k" | grep -oE "Mean KLD: +[0-9.e-]+" | grep -oE "[0-9.e-]+$"); python3 -c "import sys; v=float('$kld' or 1); sys.exit(0 if v <= 2e-5 else 1)" || { echo "GATE: KLD $kld > 2e-5 at +$o"; return 1; }
  gate "+$o" || return 1
  echo "PASS +$o"
}
echo "== stock baseline $(date +%T)"; python3 -I $N/scripts/memoc_set.py 0 >/dev/null
LD_LIBRARY_PATH=$B $B/llama-perplexity -m $M -ngl 999 -fa on -c 512 -b 512 -ub 1 --chunks 4 -f $T/corpus.txt -ctk q4_0 -ctv q4_0 --kl-divergence-base $T/kld_memoc_base.bin > $T/memoc-base.out 2>&1; echo "base logits: $(stat -c %s $T/kld_memoc_base.bin 2>/dev/null || echo MISSING) bytes"
step 0 || { echo "stock step failed — aborting"; exit 1; }
best=0; for o in 50 100 150 200; do step $o && best=$o || break; done
echo "=== ladder result: highest PASS = +$best MHz  $(date +%T)"
echo "MEMOC LADDER DONE"
