# cjm-transcript-correction-tui

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

The minimal presentation driver for cjm-transcript-correction-core: a document-order segment walk over the shared transcript graph with chunk audio playback, held-key boundary-shift gestures, inline text correction, and review markers — the human pass-1/pass-2 correction loop's UI. Kept a separate library from the headless core per the non-minimal-drivers-separate-libs rule; every correction write appends through to the workflow's sidecar journal.

## Modules

- **`cjm_transcript_correction_tui.__init__`** — Keyboard-first TUI driver for the transcript-correction workflow — a presentation
- **`cjm_transcript_correction_tui.app`**
- **`cjm_transcript_correction_tui.audio`**
- **`cjm_transcript_correction_tui.cli`**
- **`cjm_transcript_correction_tui.spine`**

## API

### `cjm_transcript_correction_tui.app`

- `CorrectionApp` _class_ — The correction loop, v0 thinnest slice: document-order segment walk with
- `load_tui_state` _function_ — Read the per-graph TUI sidecar state (last-focused positions).
- `save_tui_state` _function_ — Merge one source's last-focused position into the sidecar state file.

### `cjm_transcript_correction_tui.audio`

- `ChunkPlayer` _class_ — Persistent-output-stream VAD-chunk player — the focus-walk auto-play engine.
- `load_chunk` _function_ — Read one VAD chunk's samples from the model-input WAV — frame-sliced, sample-accurate.
- `stretch` _function_ — Pitch-preserving time-stretch (WSOLA, numpy-only) — the playback-speed engine.

### `cjm_transcript_correction_tui.cli`

- `build_parser` _function_ — The TUI driver's argument surface (mirrors correction-core's run/review args).
- `main` _function_ — Parse args, run the correction loop (the app owns the event loop + teardown).

### `cjm_transcript_correction_tui.spine`

- `ChunkRef` _class_ — Where one Segment's VAD-chunk audio lives: the model-input WAV + the chunk-local span.
- `SpineView` _class_ — One Source's effective correction spine, cursor-windowed for the TUI.
- `plan_boundary_shift` _function_ — Plan a ONE-WORD boundary shift (the [ / ] gesture unit).

## Dependencies

**Depends on:** `cjm-transcript-correction-core`, `numpy`, `sounddevice`, `soundfile`, `textual`
