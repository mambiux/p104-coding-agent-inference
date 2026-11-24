#!/usr/bin/env python3
"""fixloop.py <key> <start.html> [rounds] — generate→run→feed errors→fix loop against the llama-server on 127.0.0.1:8097 (tunnel)."""
import json, sys, subprocess, re, time, urllib.request
key, start = sys.argv[1], sys.argv[2]; rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 4; base = int(sys.argv[5]) if len(sys.argv) > 5 else 0
W = "/home/<user>/attic/longshot"; PW = "/home/<user>/.claude/jobs/44d7028b/tmp/pw"
prompt = open(f"{W}/prompt.md").read(); log = open(f"{W}/{key}-fixloop.log", "a"); results = []
def say(m): print(m, flush=True); log.write(time.strftime("%H:%M:%S ") + m + "\n"); log.flush()
def harness(path):
    out = subprocess.run(["node", "runtime-test.mjs", path], cwd=PW, capture_output=True, text=True, timeout=300).stdout.strip().splitlines()
    return json.loads(out[-1])
def clean(r):
    return r.get("total_errors") == 0 and r.get("task_add") == "shows" and r.get("task_persist") == "persists" and str(r.get("algo","")).startswith("correct") \
        and r.get("theme_toggle") == "switches" and r.get("canvas_drawn") == "drawn" and r.get("clock") == "ticks" and r.get("mobile_overflow") == "ok" and r.get("rtl") == "ltr"
def report(r):
    lines = ["I loaded your longshot.html in a real browser (Chromium) and tested it. Results:"]
    if r.get("error_samples"): lines.append("Uncaught JavaScript errors / console errors:"); lines += [f"  - {e}" for e in r["error_samples"]]
    else: lines.append("No JavaScript errors were thrown.")
    checks = []
    if r.get("rtl") != "ltr": checks.append("the whole page renders right-to-left (document direction is not ltr)")
    if r.get("clock") != "ticks": checks.append("the live clock does not update every second")
    if r.get("canvas_drawn") != "drawn": checks.append("the <canvas> chart is blank (nothing is drawn on it)")
    if r.get("task_add") != "shows": checks.append("after typing a task and submitting it (Enter and the Add button), the task never appears in the list")
    elif r.get("task_persist") != "persists": checks.append("the task disappears after reloading the page (localStorage persistence is not working)")
    if r.get("task_filter") == "none": checks.append("there is no filter control (All / Active / Completed) for the task list")
    if not str(r.get("algo","")).startswith("correct"): checks.append("entering n = 10 in the algorithm demo (Enter and the button) did not show the 10th prime, 29, anywhere on the page" + (" (it answered 11 — the brief asks for the n-th prime)" if "11" in str(r.get("algo","")) else ""))
    if r.get("theme_toggle") != "switches": checks.append("clicking the theme toggle button does not change the page background (no visible light/dark switch)" if r.get("theme_toggle") != "no button" else "there is no theme toggle button")
    if r.get("mobile_overflow") != "ok": checks.append("at 360px width the page scrolls horizontally")
    if checks: lines.append("Failed requirements:"); lines += [f"  - {c}" for c in checks]
    lines.append("\nFix every item above without dropping any other requirement from the brief. Reply with ONLY the complete corrected HTML document in a single ```html code block.")
    return "\n".join(lines)
html = open(start).read()
for rnd in range(base, base + rounds + 1):
    path = f"{W}/{key}-fix{rnd}.html"; open(path, "w").write(html)
    r = harness(path); r["round"] = rnd; results.append(r); say(f"round {rnd}: errors={r.get('total_errors')} task={r.get('task_add')}/{r.get('task_persist')} algo={r.get('algo')} theme={r.get('theme_toggle')} canvas={r.get('canvas_drawn')} clean={clean(r)}")
    json.dump(results, open(f"{W}/{key}-fixloop.json", "w"), indent=1)
    if clean(r): say("CLEAN — stopping"); break
    if rnd == base + rounds: say("budget of rounds exhausted"); break
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": "```html\n" + html + "\n```"}, {"role": "user", "content": report(r)}]
    body = {"messages": msgs, "max_tokens": int(sys.argv[4]) if len(sys.argv) > 4 else 9000, "temperature": float(sys.argv[7]) if len(sys.argv) > 7 else 0.3, "top_p": 0.95}
    if len(sys.argv) > 8: body["top_k"] = int(sys.argv[8])
    if len(sys.argv) > 6 and sys.argv[6] == "nothink": body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time(); req = urllib.request.Request("http://127.0.0.1:8097/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=2400)); dt = time.time() - t0
    txt = d["choices"][0]["message"].get("content") or ""; rz = d["choices"][0]["message"].get("reasoning_content") or ""; u = d.get("usage", {}); tim = d.get("timings", {})
    open(f"{W}/{key}-fix{rnd+1}.raw.txt", "w").write(txt)
    mm = re.search(r"```html\s*(.*?)```", txt, re.S) or re.search(r"(<!DOCTYPE html.*</html>)", txt, re.S | re.I)
    say(f"  fix {rnd+1}: {dt:.0f}s, {u.get('completion_tokens')} tokens, {tim.get('predicted_per_second',0):.1f} tok/s, finish={d['choices'][0].get('finish_reason')}, code block={bool(mm)}, reasoning_chars={len(rz)}, prompt_tokens={u.get('prompt_tokens')}")
    results[-1].update({"fix_seconds": round(dt), "fix_tokens": u.get("completion_tokens"), "fix_tok_s": round(tim.get("predicted_per_second", 0), 1), "finish": d["choices"][0].get("finish_reason")})
    if not mm: say("  no HTML returned; stopping"); break
    html = mm.group(1).strip()
say("DONE")
