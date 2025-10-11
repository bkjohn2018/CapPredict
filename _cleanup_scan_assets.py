import pathlib
root=pathlib.Path('.')
assets=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ['.png','.jpg','.jpeg','.csv','.parquet','.svg']]
text_exts={'.md','.py','.txt','.rst','.json','.yaml','.yml','.ini'}
text_files=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in text_exts and not any(seg in p.parts for seg in ['.git','venv','.venv','__pycache__'])]
blob='\n'.join((t.read_text(encoding='utf-8', errors='ignore') for t in text_files))
for a in assets:
    if str(a) not in blob and a.name not in blob:
        print(f'UNREF_ASSET,{a}')
