import os, re, pathlib
root = pathlib.Path('.')
candidates=[]
for p in root.rglob('*.py'):
    if any(seg in p.parts for seg in ['.venv','venv','.git','site-packages','__pycache__','build','dist','.pytest_cache']):
        continue
    try:
        txt=p.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        continue
    lines = txt.splitlines()
    body = []
    in_triple = False
    triple_char = ''
    for ln in lines:
        s = ln.strip()
        if not in_triple and (s.startswith('"""') or s.startswith("'''")):
            in_triple = True
            triple_char = '"""' if s.startswith('"""') else "'''"
            # handle single-line triple quotes
            if s.count(triple_char) >= 2:
                in_triple = False
            continue
        elif in_triple:
            if triple_char in s:
                in_triple = False
            continue
        if s and not s.startswith('#'):
            body.append(s)
    code = body
    if len(code)==0 or all(re.fullmatch(r'(pass|from __future__ import annotations)', ln.strip()) for ln in code):
        candidates.append(('empty', str(p)))
print('\n'.join(f'EMPTY,{p}' for _,p in candidates))
