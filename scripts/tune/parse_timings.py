import json,sys
d=json.load(sys.stdin); t=d.get("timings",{})
print(f'gen={t.get("predicted_per_second",0):.1f} tok/s pp={t.get("prompt_per_second",0):.0f} n={t.get("predicted_n")} draft_n={t.get("draft_n","-")} draft_acc={t.get("draft_n_accepted","-")}')
