import ast, os, pathlib
root=pathlib.Path('.')
files=[p for p in root.rglob('*.py') if not any(seg in p.parts for seg in ['.venv','venv','.git','__pycache__','site-packages'])]
mod_by_path={}
for p in files:
    parts=list(p.parts)
    if 'src' in parts:
        idx=parts.index('src')
        mod_parts=parts[idx+1:]
    else:
        mod_parts=parts
    rel='.'.join(pathlib.Path(*mod_parts).with_suffix('').as_posix().split('/'))
    rel=rel.lstrip('.')
    mod_by_path[p]=rel
imports=set()
for p in files:
    try:
        src=p.read_text(encoding='utf-8', errors='ignore')
        tree=ast.parse(src)
    except Exception:
        continue
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imports.add(a.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                imports.add(n.module.split('.')[0])
unused=[]
for p, mod in mod_by_path.items():
    base=os.path.basename(p)
    parts=[q for q in p.parts]
    if base in ('__init__.py','main.py','__main__.py'):
        continue
    if 'tests' in parts or any('test' in prt.lower() for prt in parts):
        continue
    top=mod.split('.')[0]
    if top and top not in imports:
        if base.startswith('_cleanup_scan_'):
            continue
        unused.append(str(p))
print('\n'.join(f'UNUSED,{p}' for p in unused))
