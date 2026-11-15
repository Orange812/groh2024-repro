"""Compare text outputs of an executed notebook against the author's stored outputs."""
import json, sys, re, difflib
orig = json.load(open(sys.argv[1]))
new = json.load(open(sys.argv[2]))
verbose = len(sys.argv) > 3

def texts(cell):
    out = []
    for o in cell.get('outputs', []):
        t = o.get('output_type')
        if t == 'stream':
            out.append(('stream:' + o['name'], ''.join(o['text'])))
        elif t in ('execute_result', 'display_data'):
            d = o.get('data', {})
            if 'text/html' in d:
                out.append(('html', ''.join(d['text/html'])))
            elif 'text/plain' in d:
                out.append(('plain', ''.join(d['text/plain'])))
            if 'image/png' in d:
                out.append(('png', '<image>'))
        elif t == 'error':
            out.append(('error', o['ename']))
    return out

same = diff = 0
diffs = []
for i, (a, b) in enumerate(zip(orig['cells'], new['cells'])):
    if a['cell_type'] != 'code':
        continue
    ta, tb = texts(a), texts(b)
    # matplotlib object reprs contain memory addresses; normalise them
    def norm(s):
        s = re.sub(r'0x[0-9a-f]+', '0x?', s)
        # statsmodels summary() stamps the run date/time (text and HTML); blank them out
        s = re.sub(r'[A-Z][a-z]{2}, \d{2} [A-Z][a-z]{2} \d{4}', '<date>', s)
        s = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', '<time>', s)
        return s
    if [(k, norm(v)) for k, v in ta] == [(k, norm(v)) for k, v in tb]:
        same += 1
    else:
        diff += 1
        diffs.append(i)
        if verbose:
            print(f"\n######## cell [{i}] differs")
            sa = '\n'.join(f'[{k}] {v}' for k, v in ta).splitlines()
            sb = '\n'.join(f'[{k}] {v}' for k, v in tb).splitlines()
            for line in list(difflib.unified_diff(sa, sb, 'author', 'rerun', lineterm='', n=1))[:60]:
                print(line)
print(f"\ncode cells identical: {same}, differing: {diff} -> {diffs}")
