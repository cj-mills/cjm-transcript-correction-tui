# cjm-transcript-correction-tui

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

The minimal presentation driver for cjm-transcript-correction-core: a document-order segment walk over the shared transcript graph with chunk audio playback, held-key boundary-shift gestures, inline text correction, and review markers — the human pass-1/pass-2 correction loop's UI. Kept a separate library from the headless core per the non-minimal-drivers-separate-libs rule; every correction write appends through to the workflow's sidecar journal.

## Modules

- **`cjm_transcript_correction_tui.__init__`** — Keyboard-first TUI driver for the transcript-correction workflow — a presentation
- **`cjm_transcript_correction_tui.app`**
- **`cjm_transcript_correction_tui.cli`**
- **`cjm_transcript_correction_tui.spine`**

## API

### `cjm_transcript_correction_tui.app`

- `CorrectionApp` _class_ — The correction loop, v0 thinnest slice: document-order segment walk with
- `load_tui_state` _function_ — Read the per-graph TUI sidecar state (last-focused positions).
- `save_tui_state` _function_ — Merge one source's view state into the sidecar state file.
- `selector_for_spine` _function_ — The selector value a picker choice persists (pure): the full skeleton
- `spine_label` _function_ — One picker row's config summary for a skeleton spine (pure).

### `cjm_transcript_correction_tui.cli`

- `build_parser` _function_ — The TUI driver's argument surface (mirrors correction-core's run/review args).
- `main` _function_ — Parse args, run the correction loop (the app owns the event loop + teardown).

### `cjm_transcript_correction_tui.spine`

- `ChunkRef` _class_ — Where one Segment's VAD-chunk audio lives: the model-input WAV + the chunk-local span.
- `SeamRef` _class_ — A source-coordinate audio span across one fine-spine boundary (the g/G
- `SpineView` _class_ — One Source's effective correction spine, cursor-windowed for the TUI.
- `list_sources` _function_ — Enumerate the graph's Source nodes (the discovery corpus, 2ce81638).
- `load_source_slice` _function_ — Decode a source-coordinate slice of the ORIGINAL media to playable samples.
- `match_sources` _function_ — The --source selector (pure; shared by direct open and the picker's seed).
- `open_stack` _function_ — Bootstrap the graph capability stack, resolving the db path (2ce81638).
- `parse_entity_input` _function_ — Parse the new-speaker editor line (pure). A leading `?` marks the entity
- `parse_mark_input` _function_ — Parse the M-editor mark grammar (pure; the DEC 2a231843 TUI gesture).
- `plan_boundary_shift` _function_ — Plan a ONE-WORD boundary shift (the [ / ] gesture unit).
- `plan_chunk_insert` _function_ — Plan a chunk insertion into the seam after the cursor (the i gesture unit; pure).
- `plan_chunk_split` _function_ — Plan a chunk split at a caret position (the S gesture unit; pure).
- `plan_time_nudge` _function_ — Plan a boundary-time nudge (the ,/. and </> gesture unit; pure).
- `resolve_mark_class_token` _function_ — Resolve a leading digit token to its menu class (the M picker; pure).
- `source_status` _function_ — Correction-status-at-a-glance for one Source (the picker's detail row).

## Dependencies

**Depends on:** `cjm-context-graph-layer`, `cjm-substrate-tui-kit`, `cjm-transcript-correction-core`, `numpy`, `sounddevice`, `soundfile`, `textual`
