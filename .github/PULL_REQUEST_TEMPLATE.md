Title: chore(cleanup): remove empty/unused modules and orphaned files

Summary
- Remove legacy `CapPredict-1/` tree (duplicate, excluded by pytest).
- Delete orphaned solutions: `CapPredict.sln` and legacy `CapPredict-1/CapPredict.sln`.
- Drop empty stubs: `src/cappredict/CapPredict.py`, `src/cappredict/data_loader.py`.
- Keep package `__init__.py` files and any files referenced by substantive docs.
- Add `.cleanup_report.md` with evidence and rationale.

Evidence
- See `.cleanup_report.md` for:
  - Empty Python files inventory
  - Import-graph based unused module candidates (legacy tree only)
  - Orphaned build/config files
  - Unreferenced assets scan results

Risk Assessment
- Low risk: Only legacy duplicates and empty stubs removed.
- Tests cover core modules: data loading/generation, preprocessing, models, logging, and visualization.
- No runtime imports reference deleted files; `pytest` passes end-to-end.

Validation
- Tests: `pytest -q` passed locally.
- Linting: ruff not available in environment; no import errors observed.
- Entry points: package-level imports `import cappredict` succeed (see tests).

Follow-ups
- Align `README.md` and `docs/SETUP.md` structure diagrams with current `src/cappredict` layout and remove remaining references to Visual Studio `.sln`.
- Optionally drop `src/cappredict/CapPredict.pyproj` if VS integration is no longer desired.
