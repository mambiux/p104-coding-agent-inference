import json, re, subprocess, os, sys, shutil
base = '/home/<user>/attic/longshot/cmp'
for model in ('bonsai', 'qwen'):
    for task in ('A', 'B'):
        f = f'{base}/{model}-{task}.json'
        if not os.path.exists(f): print(model, task, 'MISSING'); continue
        j = json.load(open(f)); c = j['choices'][0]['message'].get('content') or ''
        if task == 'A':
            blocks = re.findall(r"```python\s*(.*?)```", c, re.S)
            d = f'{base}/run-{model}'; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
            if len(blocks) >= 2:
                open(f'{d}/lru.py', 'w').write(blocks[0]); open(f'{d}/test_lru.py', 'w').write(blocks[1])
                try:
                    r = subprocess.run([sys.executable, '-I', '-m', 'pytest', '-q', '-x', '--timeout=60', 'test_lru.py'] if False else [sys.executable, '-m', 'pytest', '-q', 'test_lru.py'], cwd=d, capture_output=True, text=True, timeout=120)
                    tail = (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else 'no output'
                except subprocess.TimeoutExpired:
                    tail = 'TIMEOUT (likely a deadlock in the thread-safety test or cache)'
                print(f'{model} A: {len(blocks)} blocks, module {blocks[0].count(chr(10))} lines, tests {blocks[1].count(chr(10))} lines → pytest: {tail}')
            else:
                print(f'{model} A: only {len(blocks)} python block(s) — cannot run'); print('   head:', c[:200].replace('\n', ' '))
        else:
            print(f'{model} B ({len(c)} chars): ' + re.sub(r'\s+', ' ', c)[:600])
