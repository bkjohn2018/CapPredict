import pathlib
root=pathlib.Path('.')
solns=list(root.rglob('*.sln'))
csprojs=list(root.rglob('*.csproj'))
for s in solns:
    if not csprojs:
        print(f'ORPHANED_SLN,{s}')
