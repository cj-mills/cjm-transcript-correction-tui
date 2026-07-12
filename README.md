# cjm-transcript-correction-tui

Keyboard-first TUI driver for the transcript-correction workflow: a cursor-windowed, document-order segment walk with VAD-chunk audio auto-play, fidelity edits, boundary-shift corrections, and agents-propose/humans-confirm triage — a presentation driver bound onto `cjm-transcript-correction-core`'s operation vocabulary (the core stays headless; this library owns interaction only).

**This is the first library born on-graph**: package content is authored as graph nodes (symbols + text regions) in the self-hosting dev graph and projected to `.py` files, not hand-edited. This scaffold commit carries infrastructure only — the package arrives via graph emission.

- Note: `version` is static in `pyproject.toml` until the graph-projected `__init__.py` lands (then it flips to `dynamic`).
