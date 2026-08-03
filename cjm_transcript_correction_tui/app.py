import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_context_graph_layer.journal import sidecar_journal_path
from cjm_substrate.core.workspace import resolve_workspace
from cjm_substrate_tui_kit.audio import ChunkPlayer, load_chunk, stretch
from cjm_substrate_tui_kit.state import SidecarState
from cjm_transcript_correction_core.graph import (commit_boundary_shift_correction,
                                                  commit_chunk_insert_correction,
                                                  commit_chunk_insert_removal,
                                                  commit_chunk_split_correction,
                                                  commit_chunk_split_removal,
                                                  commit_extraction_gate, commit_mark_correction,
                                                  commit_mark_dismissal, commit_prune_amendment,
                                                  commit_speaker_assign_correction,
                                                  commit_speaker_entity,
                                                  commit_speech_overlay_correction,
                                                  commit_speech_overlay_removal,
                                                  commit_text_correction,
                                                  commit_time_nudge_correction,
                                                  fa_words_for_transcript, LEGACY_SKELETON,
                                                  list_source_spines, list_speaker_entities,
                                                  session_purposes_by_source, start_session)
from cjm_transcript_correction_core.models import (RECOMMENDED_INSERT_LABELS,
                                                   RECOMMENDED_MARK_CLASSES,
                                                   RECOMMENDED_OVERLAY_LABELS)
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, Static

from .spine import (list_sources, load_source_slice, match_sources, open_stack, parse_entity_input,
                    parse_mark_input, plan_boundary_shift, plan_chunk_insert, plan_chunk_split,
                    plan_gate, plan_time_nudge, resolve_mark_class_token, segment_word_tokens,
                    snap_word_span, source_status, SpineView)


class CorrectionApp(App):
    """The correction loop, v0 thinnest slice: document-order segment walk with
    VAD-chunk auto-play and in-place fidelity edits, over the shared transcription
    graph through correction-core's operation vocabulary.

    Interaction contract (DEC 54640079 + the walkthrough capture): the surface is a
    CENTER-PINNED window over the cursor-parameterized effective spine (drive
    round 4 ratification): the focused card's text line sits at the exact screen
    center, neighbor cards stack outward and absorb the varying text heights, so
    the eyes never leave center — segments flow past the pin. Scrolling (keys AND
    wheel) moves the CURSOR, the paint recomposes around it, nothing moves
    unbidden, content never overlaps. Focusing a segment auto-plays its VAD chunk
    from the model-input WAV (immediate-play; churn accepted per the spike). An
    edit commits a `text_content` Correction (+ its REVIEWED marker) and updates
    the local effective text — decisions persist, the worklist stays derived.
    The graph stack opens INSIDE the app (`on_mount`) so the JobQueue lives on
    Textual's event loop."""

    AUTO_FOCUS = None  # the hidden editor Input must not swallow the walk keys at mount

    SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)  # the [ ] playback-rate ladder (0.5/3.0 = the comprehension bounds, drive-round-7 verdict)
    WORDLESS_INSERT_LABELS = {"inhale", "empty", "throat-clear", "background-noise",
                              "click", "background-music", "background-voices", "echo",
                              "wheeze"}  # insert classes never meant to carry words (shift-across hop + z fold candidates; DEC a5754fa4; 'empty' = the sole silence term ('dead-air' retired, 8c0aa0bf); background-noise/click adopted 2026-07-30 (phenomenon-true FP labels — bench derives RELABELED, training gets explicit hard negatives); empty-text guard backstops all

    NUDGE_STEPS_MS = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)  # the { } nudge-step ladder (first drive: 100ms fits some cuts, others need 20/10/5; 1ms added 2026-07-27 — a Chris Wright boundary at ~358.5s outgrew 5ms — granularity is per-BOUNDARY)

    NUDGE_TAIL_S = 2.0  # Max seconds of segment TAIL an end-nudge replays (the edge under judgment, not the whole segment)

    # Lane vocabulary (DEC cc55a7b5 multi-lane workbench, v1 = walk + assign per
    # 8a4df244): the assign lane exposes ONLY these actions; assign-only actions
    # are inert in the walk lane. One data table, one check_action gate.
    ASSIGN_LANE_ACTIONS = frozenset({
        "next", "prev", "replay", "seam_next", "seam_prev", "speed_down", "speed_up",
        "yank", "assign_pick", "assign_same", "assign_new", "assign_accept",
        "cycle_lane", "cycle_lane_prev", "cancel", "quit_app"})
    ASSIGN_ONLY_ACTIONS = frozenset({"assign_pick", "assign_same", "assign_new",
                                     "assign_accept"})

    # The propose lane (leg 4, DEC 8e05b87b): model event proposals driven
    # through the SAME accept-is-an-insert-op gesture; nudges on top are the
    # edit record, skipping past is the unmarked reject. Manual insert keys
    # stay live — a model miss found by ear is bench data too.
    PROPOSE_LANE_ACTIONS = frozenset({
        "next", "prev", "replay", "seam_next", "seam_prev", "speed_down", "speed_up",
        "yank", "nudge_end_earlier", "nudge_end_later", "nudge_start_earlier",
        "nudge_start_later", "nudge_step_down", "nudge_step_up",
        "insert_chunk", "insert_labeled", "relabel_insert", "remove_insert", "edit",
        "propose_accept", "propose_next", "propose_prev", "propose_audition",
        "toggle_tier2", "cycle_lane", "cycle_lane_prev", "cancel", "quit_app"})
    PROPOSE_ONLY_ACTIONS = frozenset({"propose_accept", "propose_next", "propose_prev",
                                      "propose_audition", "toggle_tier2"})

    # The annotate lane (check fc42614d, DEC 4e05a066): TEXT-INDEXED word-span
    # sample creation — the word cursor walks the focused segment's words, v
    # anchors a range, and a label commit derives the TIME span from FA word
    # timestamps (snap-to-word; estimation fallback). A dedicated lane so
    # sample drives never pollute the walk or propose vocabularies.
    ANNOTATE_LANE_ACTIONS = frozenset({
        "next", "prev", "replay", "seam_next", "seam_prev", "speed_down", "speed_up",
        "yank", "word_left", "word_right", "word_select", "annotate_quick",
        "annotate_pick", "annotate_editor", "annotate_audition", "overlay_remove",
        "overlay_nudge", "nudge_step_down", "nudge_step_up",
        "next_overlay", "prev_overlay", "toggle_wordless_fold",
        "cycle_lane", "cycle_lane_prev", "cancel", "quit_app"})
    ANNOTATE_ONLY_ACTIONS = frozenset({
        "word_left", "word_right", "word_select", "annotate_quick", "annotate_pick",
        "annotate_editor", "annotate_audition", "overlay_remove", "overlay_nudge",
        "next_overlay", "prev_overlay"})

    CSS = """
    #cards { height: 1fr; overflow: hidden hidden; }
    """

    BINDINGS = [
        Binding("j", "next", "next"),
        Binding("down", "next", "next", show=False),
        Binding("s", "next", "next", show=False),
        Binding("k", "prev", "prev"),
        Binding("up", "prev", "prev", show=False),
        Binding("w", "prev", "prev", show=False),
        Binding("r", "replay", "replay"),
        Binding("g", "seam_next", "seam audio"),
        Binding("G", "seam_prev", "seam audio ←", show=False),
        Binding("comma", "nudge_end_earlier", "nudge −", key_display=","),
        Binding("full_stop", "nudge_end_later", "nudge +", key_display="."),
        Binding("less_than_sign", "nudge_start_earlier", "start −", show=False),
        Binding("greater_than_sign", "nudge_start_later", "start +", show=False),
        Binding("comma", "overlay_nudge('end', -1)", show=False),
        Binding("full_stop", "overlay_nudge('end', 1)", show=False),
        Binding("less_than_sign", "overlay_nudge('start', -1)", show=False),
        Binding("greater_than_sign", "overlay_nudge('start', 1)", show=False),
        Binding("left_curly_bracket", "nudge_step_down", "step −", show=False, key_display="{"),
        Binding("right_curly_bracket", "nudge_step_up", "step +", show=False, key_display="}"),
        Binding("left_square_bracket", "speed_down", "slower", key_display="["),
        Binding("right_square_bracket", "speed_up", "faster", key_display="]"),
        Binding("i", "insert_chunk", "insert chunk"),
        Binding("I", "insert_labeled", "insert+label", show=False),
        Binding("L", "relabel_insert", "relabel insert", show=False),
        Binding("x", "remove_insert", "remove insert", show=False),
        Binding("S", "split_chunk", "split chunk", show=False),
        Binding("e", "edit", "edit text"),
        Binding("y", "yank", "copy text"),
        Binding("right", "shift_push", "push word", key_display="→"),
        Binding("d", "shift_push", "push word", show=False),
        Binding("left", "shift_pull", "pull word", key_display="←"),
        Binding("a", "shift_pull", "pull word", show=False),
        Binding("tab", "cycle_lane", "lane", show=False, priority=True),
        Binding("shift+tab", "cycle_lane_prev", "lane ←", show=False, priority=True),
        Binding("space", "assign_same", "same speaker", show=False),
        Binding("A", "assign_new", "new speaker", show=False),
        Binding("a", "assign_accept", "accept cluster", show=False),
        Binding("1", "assign_pick(1)", show=False), Binding("2", "assign_pick(2)", show=False),
        Binding("3", "assign_pick(3)", show=False), Binding("4", "assign_pick(4)", show=False),
        Binding("5", "assign_pick(5)", show=False), Binding("6", "assign_pick(6)", show=False),
        Binding("7", "assign_pick(7)", show=False), Binding("8", "assign_pick(8)", show=False),
        Binding("9", "assign_pick(9)", show=False),
        Binding("a", "propose_accept", "accept proposal", show=False),
        Binding("n", "propose_next", "next proposal", show=False),
        Binding("N", "propose_prev", "prev proposal", show=False),
        Binding("R", "propose_audition", "audition proposal", show=False),
        Binding("t", "toggle_tier2", "tier-2 audition", show=False),
        Binding("h", "word_left", "word ←", show=False),
        Binding("l", "word_right", "word →", show=False),
        Binding("left", "word_left", show=False),
        Binding("right", "word_right", show=False),
        Binding("v", "word_select", "select words", show=False),
        Binding("space", "annotate_quick", show=False),
        Binding("A", "annotate_editor", show=False),
        Binding("R", "annotate_audition", show=False),
        Binding("x", "overlay_remove", show=False),
        Binding("n", "next_overlay", show=False),
        Binding("N", "prev_overlay", show=False),
        Binding("1", "annotate_pick(1)", show=False), Binding("2", "annotate_pick(2)", show=False),
        Binding("3", "annotate_pick(3)", show=False), Binding("4", "annotate_pick(4)", show=False),
        Binding("5", "annotate_pick(5)", show=False), Binding("6", "annotate_pick(6)", show=False),
        Binding("7", "annotate_pick(7)", show=False), Binding("8", "annotate_pick(8)", show=False),
        Binding("9", "annotate_pick(9)", show=False),
        Binding("m", "mark_quick", "mark"),
        Binding("b", "mark_boundary", "mark boundary"),
        Binding("M", "mark_editor", "mark+class"),
        Binding("n", "next_mark", "next mark"),
        Binding("N", "prev_mark", "prev mark"),
        Binding("p", "next_prune", "next prune"),
        Binding("P", "prev_prune", "prev prune"),
        Binding("F", "gate_editor", "flywheel gate", show=False),
        Binding("enter", "open_source", "open", show=False),
        Binding("z", "toggle_wordless_fold", "fold ⊕wordless", show=False),
        Binding("escape", "cancel", "cancel/stop", show=False, priority=True),
        Binding("q", "quit_app", "quit"),
    ]

    def __init__(self, graph_db_path: Optional[str] = None,  # The shared transcription graph db (None = workspace-resolved, 2ce81638)
                 *, source: Optional[str] = None,         # Source id or title substring
                 manifests_dir: str = ".cjm/manifests",   # Capability manifests directory
                 rendition: Optional[str] = None,         # Rendition selector (None = auto)
                 skeleton: Optional[str] = None,          # Skeleton-spine selector ("legacy" | hash prefix; None = picker/sidecar decides)
                 actor: str = "human",                    # Actor recorded on corrections
                 autoplay: bool = True,                   # Auto-play the focused chunk
                 audio_device: Optional[object] = None,   # Output device (None = system default)
                 resume: bool = True,                     # Reopen at the source's last-focused segment
                 shift_floor_s: float = 0.0,              # Min seconds between held-key boundary shifts (0 = ungoverned; the commit guard is the real governor)
                 nudge_step_ms: Optional[float] = None,   # Boundary time-nudge step per ,/. press; None = sidecar-persisted preference, else 100 (the { } ladder adjusts live)
                 lane: Optional[str] = None,              # Starting pass lane ("walk" | "assign" | "annotate"); None = sidecar-persisted preference, else walk (DEC 8a4df244)
                 purpose: Optional[str] = None,           # None = genuine pass; "feature-test" tags the session excludable from flywheel datasets (--test, DEC c86714a4)
                 fa_cache_db: Optional[str] = None):      # Forced-alignment cache db (word times, the annotate lane's snap source); None = workspace-resolved
        super().__init__()
        self._open_kwargs = dict(source=source, manifests_dir=manifests_dir,
                                 rendition=rendition, skeleton=skeleton)
        self._spines: List[Dict[str, Any]] = []      # coexisting skeleton spines of the source being opened
        self._spine_source: Optional[Tuple[str, str]] = None  # (source_id, title) awaiting a spine choice
        self._graph_db_path = graph_db_path
        # Every correction write appends through to the db's sidecar journal (DEC
        # ccbab9f5); the path derives from the EFFECTIVE db at mount (may be
        # workspace-resolved, so it cannot be computed here).
        self._journal_path: Optional[object] = None
        self.stage = "select"            # "select" (source picker) -> "correct" (the walk)
        self._graph_cap = "cjm-capability-graph-sqlite"
        self._manager = None             # the open stack; view.close() owns teardown once a spine opens
        self._queue = None
        self._sources: List[Tuple[str, str]] = []     # [(source_id, title)] the picker walks
        self._status: Dict[str, Dict[str, int]] = {}  # source_id -> status-at-a-glance
        self._purposes: Dict[str, Dict[str, int]] = {}  # source_id -> session-purpose mix (d915d545 a)
        self.view: Optional[SpineView] = None
        self.player: Optional[ChunkPlayer] = None
        self.cursor = 0
        self.actor = actor
        self.purpose = purpose
        self.autoplay = autoplay
        self.speed = 1.0                   # playback rate ([ ] preset ladder; sidecar-persisted preference)
        self.audio_device = audio_device
        self.session_id: Optional[str] = None
        self._marks: Dict[int, str] = {}   # cursor position -> local decision echo
        self._mark_class = "suspect"       # last-used ⚑ class (m/b repeat it; sidecar-persisted)
        self._insert_label = "inhale"      # last-used ⊕ insert label (I pre-fills it; sidecar-persisted)
        self._overlay_label = "hesitation-marker"  # last-used ◈ overlay label (space repeats it; sidecar-persisted)
        self._word_cursor = 0              # annotate lane: word index on the focused segment
        self._word_anchor: Optional[int] = None  # annotate lane: selection anchor (None = no range; v sets)
        self._fa_cache_arg = fa_cache_db   # explicit --fa-cache-db (wins over the workspace default)
        self._fa_cache_db: Optional[Path] = None  # resolved FA cache (None = estimation-only snapping)
        self._fa_words_cache: Dict[str, Optional[List[Dict[str, Any]]]] = {}  # transcript id -> FA words (per-open memo)
        self._input_mode = "edit"          # what the hidden Input commits ("edit" | "mark" | "insert" | "assign" | "split" | "propose_split" | "gate")
        self._pending_proposal = None      # (anchor index, proposal dict) awaiting the propose-split editor hop
        self._ticker = None                # live playback-position timer (the r/R line-up readout)
        self._tick_info = None             # (t0 monotonic, span start, span end, note, speed)
        self._tick_last = ""               # what the ticker last painted — its ownership receipt (drive asks 2026-07-30)
        self.lane = lane or "walk"         # active pass lane ("walk" | "assign"; tab cycles, sidecar persists)
        self._lane_arg = lane              # explicit --lane (wins over the sidecar; None = defer)
        self._entities: List[Dict[str, Any]] = []  # speaker Entity registry (graph-side, source-spanning)
        self._active_entity: Optional[str] = None  # entity id space assigns ("same speaker continues")
        self._accept_cluster: Optional[str] = None  # cluster awaiting its name (the a-gesture's editor hop; None = no accept pending)
        self._shift_busy = False           # in-flight boundary-shift commit (key-repeat throttle)
        self._last_shift = 0.0             # last completed shift (monotonic; paint-rate floor)
        self.fold_wordless = False         # z: walk/assign passes skip + collapse wordless inserts (sidecar-persisted)
        self._shift_floor = float(shift_floor_s)  # tune with tests_manual/keyrate_probe.py
        self._nudge_step_arg = nudge_step_ms  # explicit --nudge-step-ms (wins over the sidecar; None = defer)
        self._nudge_step = 0.1             # seconds per nudge press (resolved at spine open: flag > sidecar > 100ms)
        self._nudge_busy = False           # in-flight nudge commit (key-repeat throttle)
        self.resume = resume
        self._state_saved = 0.0            # last sidecar bookmark write (monotonic; 1s throttle)

    def compose(self) -> ComposeResult:
        yield Static("", id="cards")
        yield Static("loading spine…", id="status")
        editor = Input(id="editor")
        editor.display = False
        yield editor

    async def on_mount(self) -> None:
        self._manager, self._queue, db = await open_stack(
            self._graph_db_path, manifests_dir=self._open_kwargs["manifests_dir"],
            graph_capability=self._graph_cap)
        self._graph_db_path = db
        self._journal_path = sidecar_journal_path(db)
        self.player = ChunkPlayer(device=self.audio_device)
        sources = await list_sources(self._queue, self._graph_cap)
        picked = match_sources(sources, self._open_kwargs["source"])
        if len(picked) == 1:
            await self._open_source(*picked[0])
            return
        # 2ce81638 discovery: no unique --source -> browse the graph's Sources
        # (a bad needle widens to ALL of them, never a dead-end error).
        self._sources = picked if len(picked) > 1 else sources
        for sid, _ in self._sources:
            self._status[sid] = await source_status(self._queue, self._graph_cap, sid)
        # Purpose mix at a glance (d915d545 a): which sources carry only
        # dev-era feature-test edits vs genuine passes — one sessions read.
        self._purposes = await session_purposes_by_source(self._queue, self._graph_cap)
        self.cursor = 0
        self._render()

    async def _open_source(self, source_id: str, title: str) -> None:
        """Open one Source: resolve WHICH skeleton spine first (DEC f1024568).

        One spine (or an explicit --skeleton) opens directly. Coexisting spines
        ALWAYS show the picker — the sidecar choice pre-positions the cursor on
        the last-opened spine rather than auto-opening it (user 2026-07-22:
        memory = position, not a bypass; switching spines must stay one glance
        away)."""
        selector = self._open_kwargs["skeleton"]
        spines = await list_source_spines(self._queue, self._graph_cap, source_id,
                                          rendition_selector=self._open_kwargs["rendition"])
        if selector is None and len(spines) > 1:
            saved = load_tui_state(self._graph_db_path).get(source_id) or {}
            last = str(saved.get("skeleton") or "")
            self._spines = spines
            self._spine_source = (source_id, title)
            self.stage = "spine"
            self.cursor = next((i for i, sp in enumerate(spines)
                                if selector_for_spine(sp) == last), 0)
            self._render()
            return
        await self._open_spine(source_id, title, selector)

    async def _open_spine(self, source_id: str, title: str,
                          skeleton: Optional[str]) -> None:
        """Open one Source's CHOSEN spine on the already-open stack and enter the walk."""
        self.view = await SpineView.open_on(self._manager, self._queue, self._graph_cap,
                                            source_id, title,
                                            rendition=self._open_kwargs["rendition"],
                                            skeleton=skeleton)
        self.stage = "correct"
        sess = await start_session(self.view.queue, self.view.graph_id,
                                   [self.view.source_id],
                                   journal_path=self._journal_path,
                                   purpose=self.purpose)
        self.session_id = sess.id
        state = load_tui_state(self._graph_db_path)
        try:
            # Speed is a PREFERENCE, not a position — restored even with resume=False.
            self.speed = float(state.get("_speed") or 1.0)
        except (TypeError, ValueError):
            self.speed = 1.0
        # Nudge step: explicit flag > sidecar preference > 100ms (same
        # preference tier as speed; the { } ladder adjusts + persists it).
        try:
            saved_ms = float(state.get("_nudge_step_ms") or 0.0)
        except (TypeError, ValueError):
            saved_ms = 0.0
        step_ms = (float(self._nudge_step_arg) if self._nudge_step_arg is not None
                   else (saved_ms if saved_ms > 0 else 100.0))
        self._nudge_step = step_ms / 1000.0
        mc = str(state.get("_mark_class") or "suspect")
        self._mark_class = mc if mc[:1].isalnum() else "suspect"   # heal a junk-class sidecar
        il = str(state.get("_insert_label") or "inhale")
        self._insert_label = il if il[:1].isalnum() else "inhale"
        # Lane: explicit flag > sidecar preference > walk (DEC 8a4df244); the
        # speaker Entity registry loads once per open (source-spanning, people-scale).
        saved_lane = str(state.get("_lane") or "")
        self.lane = self._lane_arg or (saved_lane if saved_lane in ("walk", "assign", "annotate")
                                       else "walk")
        ol = str(state.get("_overlay_label") or "hesitation-marker")
        self._overlay_label = ol if ol[:1].isalnum() else "hesitation-marker"
        # The annotate lane's snap source: explicit flag > the workspace's FA
        # capability cache; missing = the lane degrades to estimated spans.
        self._fa_cache_db = self._resolve_fa_cache()
        self._fa_words_cache = {}
        self._word_cursor, self._word_anchor = 0, None
        self.fold_wordless = bool(state.get("_fold_wordless") or False)
        self._entities = await list_speaker_entities(self.view.queue, self.view.graph_id)
        self._active_entity = None
        self.cursor = 0                    # the picker borrowed the cursor
        if self.resume:
            saved = state.get(self.view.source_id)
            if saved and self.view.size:
                self.cursor = max(0, min(self.view.size - 1, int(saved.get("cursor", 0))))
        self._render()
        if self.autoplay:
            self._play_cursor()

    def on_resize(self, event) -> None:
        if self.view is not None:
            self._render()

    def _card_lines(self, pos: int, width: int) -> Tuple[List[Text], int]:
        """One segment card as styled screen lines + the offset of its first body line.

        FIXED GUTTER, ONE TEXT LANE (presentation agenda item 1): index/time/marks
        live in a fixed-width left column and ALWAYS recede (dim — the eye must be
        unable to accidentally read a timestamp); segment text gets its own
        consistently-indented lane, so walking scans a single vertical column of
        pure prose. Focus emphasis carries over: cursor±1 lane text bright, far
        field dim, the focused card a full-width reverse band."""
        view = self.view
        seg = view.segments[pos]
        gut_w = self._gutter_w
        if pos != self.cursor and self._folded(pos):
            # Folded wordless insert (z, DEC fdb93036): one dim line keeps the
            # spatial context — an inhale bookend needing a nudge stays
            # spottable — without a full card or a focus stop.
            lab = view.insert_labels.get(seg.id) or "wordless"
            row = Text()
            row.append(f"⊕{seg.index}".ljust(gut_w), style="dim cyan")
            row.append(f"({lab} · {seg.start_time:.1f}–{seg.end_time:.1f}s)"
                       if seg.start_time is not None else f"({lab})", style="dim")
            return [row], 0
        lane_w = max(10, width - gut_w)
        # Corrected-state DERIVES from committed corrections; the manual
        # reviewed verdict is retired (DEC c1bb202f — absence of edits IS the
        # no-edits signal, a stored claim of absence can only go stale).
        mark = "✎" if self._marks.get(pos) == "corrected" else "·"
        # Gutter styling must ride SPANS, not the Text base style: lane text is
        # appended onto these same row objects, and a base style would bleed
        # into it (the round-2 drive regression — first two lane lines dimmed).
        g1 = Text()
        if seg.id in view.inserted_ids:
            # A synthetic (inserted) chunk: no layer-0 index of its own — the ⊕
            # + flank index reads as "grafted after #N" (DEC 3d3fa2a8).
            g1.append(f"⊕{seg.index} {mark}", style="cyan")
        else:
            g1.append(f"#{seg.index} {mark}", style="dim")
        if seg.id in view.pruned_ids:
            g1.append(" ✂", style="red")
        if seg.id in view.marked_ids:
            g1.append(" ⚑", style="yellow")
        if seg.id in view.overlay_ids:
            g1.append(" ◈", style="cyan")
        g2 = Text()
        g2.append(f"{seg.start_time:.1f}–{seg.end_time:.1f}s"
                  if seg.start_time is not None else "(no audio)", style="dim")
        if seg.text:
            body = Text(seg.text)
        elif seg.id in view.inserted_ids:
            lab = view.insert_labels.get(seg.id)
            body = Text(f"(inserted{': ' + lab if lab else ''})", style="cyan")
        else:
            body = Text("(empty)", style="dim")
        if self.lane == "assign":
            # Attribution chip: the assign lane's object of attention rides the
            # text lane (gutter width stays source-stable). ∅ = unassigned;
            # an unassigned segment with a diarization proposal paints the
            # cluster chip (?S00-style, per-cluster tint) — what `a` accepts.
            sp = view.speakers.get(seg.id)
            prop = view.turn_proposals.get(seg.id)
            if sp:
                chip = Text(f"{self._entity_name(sp['entity_id'])[:14]} ▏", style="magenta")
            elif prop:
                chip = Text(f"?{str(prop['cluster']).replace('SPEAKER_', 'S')} ▏",
                            style=self._cluster_style(str(prop["cluster"])))
            else:
                chip = Text("∅ ▏", style="dim")
            chip.append_text(body)
            body = chip
        elif self.lane == "propose":
            # Pending-proposal chip: the propose lane's object of attention —
            # ?label + score on the ANCHOR card (the segment the accept would
            # insert after); ×n when several stack in one gap.
            props = view.event_proposals.get(seg.id)
            if props:
                p = props[0]
                extra = f"×{len(props)}" if len(props) > 1 else ""
                # ?? = audition tier (below the operating point, 3a5cb858) —
                # quieter mark, same accept gesture.
                q = "??" if int(p.get("tier", 1)) == 2 else "?"
                chip = Text(f"{q}{p.get('label')} {float(p.get('score') or 0):.2f}{extra} ▏",
                            style="dim magenta" if q == "??" else "dim cyan")
                chip.append_text(body)
                body = chip
        elif self.lane == "annotate" and pos == self.cursor and seg.text:
            body = self._annotate_body(seg)
        if abs(pos - self.cursor) > 1 and seg.text:
            body.stylize("dim")
        lane = body.wrap(self.console, lane_w)
        lines: List[Text] = []
        a = view.aseg_index(pos)
        if a is not None and (pos == 0 or view.aseg_index(pos - 1) != a):
            lines.append(Text(f"━━━ audio segment {a} ━━━", style="yellow"))
        body_offset = len(lines)
        gutter = [g1, g2]
        for i in range(max(len(gutter), len(lane))):
            row = gutter[i] if i < len(gutter) else Text("")
            row.pad_right(max(0, gut_w - row.cell_len))
            if i < len(lane):
                row.append_text(lane[i])
            lines.append(row)
        if pos == self.cursor:
            for ln in lines:
                ln.pad_right(max(0, width - ln.cell_len))
                ln.stylize("reverse")
        return lines, body_offset

    @property
    def _gutter_w(self) -> int:
        """The source-wide gutter width: sized ONCE from the last segment (the widest
        index + time span), so the text lane's indent never wobbles while walking."""
        last = self.view.segments[-1]
        t_w = (len(f"{last.end_time:.1f}–{last.end_time:.1f}s")
               if last.end_time is not None else 0)
        return max(t_w, len("(no audio)"), len(f"#{last.index}") + 6) + 2  # +6: the ✓/✂/⚑ glyph rail

    def _render(self) -> None:
        """Center-pinned paint (drive round 4): the focused card's FIRST TEXT LINE
        is pinned to the vertical center of the card area; neighbor cards stack
        outward from it (one blank separator row) and absorb the height variance,
        clipping at the screen edges. The pin never moves — the spine flows past it."""
        if self.stage == "select":
            self._render_picker()
            return
        if self.stage == "spine":
            self._render_spine_picker()
            return
        view = self.view
        if not view.size:
            self.query_one("#status", Static).update(f"{view.source_title}  ·  empty spine")
            return
        width = max(20, self.size.width)
        height = max(3, self.size.height - 1)   # the status line keeps the last row
        rows: List[Optional[Text]] = [None] * height

        def place(lines: List[Text], top: int) -> None:
            for i, ln in enumerate(lines):
                if 0 <= top + i < height:
                    rows[top + i] = ln

        f_lines, f_off = self._card_lines(self.cursor, width)
        top_f = height // 2 - f_off             # body line 0 lands dead center
        place(f_lines, top_f)
        pos, bottom = self.cursor - 1, top_f - 2
        while pos >= 0 and bottom >= 0:
            lines, _ = self._card_lines(pos, width)
            place(lines, bottom - len(lines) + 1)
            bottom -= len(lines) + 1
            pos -= 1
        pos, top = self.cursor + 1, top_f + len(f_lines) + 1
        while pos < view.size and top < height:
            lines, _ = self._card_lines(pos, width)
            place(lines, top)
            top += len(lines) + 1
            pos += 1
        self.query_one("#cards", Static).update(
            Text("\n").join(ln if ln is not None else Text("") for ln in rows))
        self.query_one("#status", Static).update(self._status_line())

    def _status_line(self) -> str:
        """The unified status strip (DEC cc55a7b5): lane badge + session-lane
        badge (d915d545 b — TEST PASS under --test, nothing when genuine) +
        position + lane-scoped counters + the ACTIVE LANE's keybar only."""
        view = self.view
        badges = {"assign": "\\[ASSIGN]", "propose": "\\[PROPOSE]",
                  "annotate": "\\[ANNOTATE]"}.get(self.lane, "\\[WALK]")
        if self.purpose:
            badges += (" \\[TEST PASS]" if self.purpose == "feature-test"
                       else f" \\[{self.purpose.upper()}]")
        head = (f"{badges}  {view.source_title}"
                f"  ·  segment {self.cursor + 1}/{view.size}")
        chip = self._gate_chip()
        if chip:
            head += f"  ·  {chip}"
        tail = f"  ·  ×{self.speed:g}  ·  session {str(self.session_id or '')[:8]}"
        if self.lane == "assign":
            assigned = sum(1 for s in view.segments if s.id in view.speakers)
            active = (self._entity_name(self._active_entity)
                      if self._active_entity else "none")
            meta = view.turns_meta.get("metadata") or {}
            turns = (f"  ·  turns {len(view.turn_proposals)}/{view.size}"
                     f" · {meta.get('speaker_count', '?')}spk"
                     if view.turn_proposals else "  ·  no turns")
            return (f"{head}  ·  assigned {assigned}/{view.size}{turns}"
                    f"  ·  speaker: {active}{tail}"
                    f"  ·  a accept · 1-9 pick · space same · A new · j/k walk · r replay"
                    f" · g/G seam · \\[/] speed · y copy · tab walk-lane · q quit")
        if self.lane == "propose":
            meta = view.proposals_meta or {}
            pending = meta.get("pending", 0)
            t2 = meta.get("tier2_total", 0)
            tier2 = (f" · tier2 {t2} {'shown' if view.show_tier2 else 'hidden'}"
                     if t2 else "")
            return (f"{head}  ·  proposals {pending} pending{tier2}"
                    f" · set {str(meta.get('proposal_set_id') or '')[-8:]}"
                    f" · model {str(meta.get('training_run_id') or '')[-8:]}{tail}"
                    f"  ·  a accept · n/N jump · R proposal{' · t tier2' if t2 else ''}"
                    f" · r chunk · ,./<> nudge"
                    f" · i/I manual · L relabel · x remove · e edit · j/k walk"
                    f" · g/G seam · tab lane · q quit")
        if self.lane == "annotate":
            seg = view.segments[self.cursor]
            toks = segment_word_tokens(seg.text)
            sel = self._selection_range(len(toks))
            if toks and sel is not None:
                a, b = sel
                readout = " ".join(t for _, _, t in toks[a:b + 1])
                readout = readout if len(readout) <= 30 else readout[:29] + "…"
                sel_txt = f"  ·  sel “{readout}”"
            else:
                sel_txt = "  ·  (no words here)" if not toks else ""
            return (f"{head}  ·  ◈ {view.overlay_count}{sel_txt}"
                    f"  ·  label: {self._overlay_label}{tail}"
                    f"  ·  h/l·←→ word · v range · space ◈commit · 1-9 class"
                    f" · A class+ · R audition · ,./<> ◈nudge · x remove · n/N ◈ jump"
                    f" · j/k walk · r replay · tab lane · q quit")
        edited = sum(1 for v in self._marks.values() if v == "corrected")
        return (f"{head}  ·  edited {edited}{tail}"
                f"  ·  j/k·w/s walk · ←→/a/d shift · r replay · g/G seam · ,./<> nudge"
                f" · {{}} step · \\[/] speed · e edit · y copy · i/I ⊕insert · x ⊖remove"
                f" · m/b/M ⚑mark · n/N⚑ p/P✂ jump · z fold⊕ · F gate · tab assign-lane · q quit")

    def _render_picker(self) -> None:
        """The 2ce81638 discovery stage: the graph's Sources with correction
        status at a glance; same key vocabulary as the walk (j/k, enter opens).
        Spans only — no base row styles (7aca1117)."""
        width = max(20, self.size.width)
        lines: List[Text] = [Text("")]
        if not self._sources:
            lines.append(Text("  no Source nodes on this graph", style="dim"))
        for i, (sid, title) in enumerate(self._sources):
            st = self._status.get(sid) or {}
            focused = (i == self.cursor)
            row = Text("")
            row.append("  > " if focused else "    ")
            row.append(title or sid[:12], style="bold" if focused else "")
            row.append(f"   {st.get('segments', 0)} segs", style="dim")
            row.append(f" · {st.get('corrections', 0)} corrections", style="dim")
            marks = st.get("marks", 0)
            if marks:
                row.append(f" · {marks} ⚑", style="yellow")
            mix = self._purposes.get(sid) or {}
            genuine = mix.get("genuine", 0)
            tests = sum(n for p, n in mix.items() if p != "genuine")
            if genuine:
                # Genuine passes are the flywheel feedstock — the count leads.
                row.append(f" · genuine: {genuine}", style="green")
                if tests:
                    row.append(f" (+{tests} test)", style="dim")
            elif tests:
                row.append(" · all test", style="yellow")
            row.truncate(width)
            lines.append(row)
        self.query_one("#cards", Static).update(Text("\n").join(lines))
        tail = str(self._graph_db_path or "")
        tail = tail if len(tail) <= 40 else "…" + tail[-39:]
        self.query_one("#status", Static).update(
            f"pick a source ({len(self._sources)})  ·  @{tail}"
            f"  ·  j/k walk · enter open · q quit")

    def _render_spine_picker(self) -> None:
        """The spine picker (DEC f1024568): one row per coexisting SKELETON —
        config summary + segment count — when a source carries more than one
        (e.g. the pre-split spine beside a sentence-split re-decomposition).
        Always shown for multi-spine sources; the sidecar choice pre-positions
        the cursor on the last-opened spine. Spans only — no base row styles
        (7aca1117)."""
        width = max(20, self.size.width)
        _, title = self._spine_source or ("", "")
        lines: List[Text] = [Text("")]
        header = Text("  ")
        header.append(title or "source", style="bold")
        header.append(f"  ·  {len(self._spines)} spines coexist — pick one", style="dim")
        lines.append(header)
        lines.append(Text(""))
        for i, sp in enumerate(self._spines):
            focused = (i == self.cursor)
            row = Text("")
            row.append("  > " if focused else "    ")
            row.append(spine_label(sp), style="bold" if focused else "")
            row.append(f"   {sp.get('segments', 0)} segs", style="dim")
            row.truncate(width)
            lines.append(row)
        self.query_one("#cards", Static).update(Text("\n").join(lines))
        self.query_one("#status", Static).update(
            "pick a spine  ·  j/k walk · enter open (choice persists) · q quit")

    def check_action(self, action: str, parameters) -> bool:
        """Stage gate + LANE gate (one data table, one gate — DEC cc55a7b5).

        During the picker only walk/open/quit act (view-None crash guard). In
        the walk, the ACTIVE LANE scopes the vocabulary: the assign lane
        exposes only ASSIGN_LANE_ACTIONS; assign-only actions are inert in the
        walk lane (lane-scoping is what makes the space overload safe,
        8a4df244)."""
        if self.stage in ("select", "spine"):
            return action in ("next", "prev", "open_source", "quit_app")
        if self.lane == "assign":
            return action in self.ASSIGN_LANE_ACTIONS
        if self.lane == "propose":
            return action in self.PROPOSE_LANE_ACTIONS
        if self.lane == "annotate":
            return action in self.ANNOTATE_LANE_ACTIONS
        return action not in (self.ASSIGN_ONLY_ACTIONS | self.PROPOSE_ONLY_ACTIONS
                              | self.ANNOTATE_ONLY_ACTIONS)

    async def action_open_source(self) -> None:
        if self.stage == "spine":
            if not self._spines or self._spine_source is None:
                return
            sid, title = self._spine_source
            selector = selector_for_spine(self._spines[self.cursor])
            # Persist the choice: it pre-positions the picker cursor next open
            # (the menu itself always shows — user 2026-07-22).
            save_tui_state(self._graph_db_path, sid, None,
                           skeleton=selector, spines=len(self._spines))
            await self._open_spine(sid, title, selector)
            return
        if self.stage != "select" or not self._sources:
            return
        sid, title = self._sources[self.cursor]
        await self._open_source(sid, title)

    def _wordless_insert(self, pos: int) -> bool:
        """A certified-wordless inserted chunk: wordless CLASS and empty text.

        Label AND text must both agree (DEC a5754fa4): an empty
        hesitation-marker is NOT wordless — its um lands later, and having
        hopped it (shift) or hidden it (fold) would misplace that word
        retroactively."""
        seg = self.view.segments[pos]
        return (seg.id in self.view.inserted_ids
                and not (seg.text or "").strip()
                and str(self.view.insert_labels.get(seg.id) or "") in self.WORDLESS_INSERT_LABELS)

    def _folded(self, pos: int) -> bool:
        """Is this position folded away right now? (z toggle; never in the
        propose lane — pending proposals may anchor to inserted segments)."""
        return (self.fold_wordless and self.lane != "propose"
                and self._wordless_insert(pos))

    def action_toggle_wordless_fold(self) -> None:
        """z: fold/unfold wordless inserts for follow-up passes (drive ask
        2026-07-30, DEC fdb93036) — folded chunks are skipped by single-step
        walking (never focused, so never auto-played) and paint as one dim
        line; explicit jumps still land anywhere. Sidecar-persisted like
        speed."""
        if self.view is None or self.stage in ("select", "spine"):
            return
        self.fold_wordless = not self.fold_wordless
        moved = ""
        if self._folded(self.cursor):
            for j in (*range(self.cursor + 1, self.view.size),
                      *range(self.cursor - 1, -1, -1)):
                if not self._folded(j):
                    self.cursor = j
                    moved = f" · cursor → #{self.view.segments[j].index}"
                    break
        save_tui_state(self._graph_db_path, self.view.source_id, self.cursor,
                       fold_wordless=self.fold_wordless)
        n = sum(1 for p in range(self.view.size) if self._folded(p))
        self._render()
        self.query_one("#status", Static).update(
            f"wordless inserts folded ({n} collapsed) · z unfolds{moved}"
            if self.fold_wordless else f"wordless inserts unfolded{moved}")

    def _start_ticker(self, start_s: float, end_s: float, note: str = "") -> None:
        """Live playback-position readout (drive ask 2026-07-29): while audio
        plays, the status line shows the CURRENT SOURCE TIME so the ear can
        line a proposed span up against the r-replay. Position derives from
        wall-clock × speed (the player has no position API); the timer
        self-terminates at span end and any new play replaces it."""
        self._stop_ticker()
        self._tick_info = (time.monotonic(), start_s, end_s, note, self.speed)
        # Claim whatever is on the line as this gesture's paint (seam-decode
        # readouts etc. precede the ticker), so the first tick may overwrite
        # it but a LATER gesture's message may not — the ownership receipt
        # `_tick` checks (drive asks 2026-07-30).
        self._tick_last = str(self.query_one("#status", Static).content)
        self._ticker = self.set_interval(0.1, self._tick)

    def _stop_ticker(self) -> None:
        if self._ticker is not None:
            self._ticker.stop()
            self._ticker = None

    def _tick(self) -> None:
        status = self.query_one("#status", Static)
        if str(status.content) != self._tick_last:
            # Another gesture painted the line mid-playback (a nudge readout,
            # a step change, a lane repaint): the ticker YIELDS — audio keeps
            # sounding, the newer message keeps the line (drive ask
            # 2026-07-30; ownership = content receipt, no caller refactor).
            self._stop_ticker()
            return
        t0, start_s, end_s, note, speed = self._tick_info
        cur = start_s + (time.monotonic() - t0) * speed
        if cur >= end_s:
            # Span done: LINGER on a final readout until the next gesture
            # (drive ask 2026-07-30 — short spans vanished before they could
            # be read; supersedes the 2026-07-29 instant hand-back, whose
            # intent survives: every gesture repaints or overwrites the
            # line, so nothing can stick).
            self._stop_ticker()
            self._tick_last = (f"■ played {start_s:.2f}–{end_s:.2f}s{note}"
                               " · any key clears")
            status.update(self._tick_last)
            return
        self._tick_last = f"▶ {cur:.2f}s · span {start_s:.2f}–{end_s:.2f}s{note} · esc stops"
        status.update(self._tick_last)

    def _play_cursor(self) -> None:
        seg = self.view.segments[self.cursor]
        note = ""
        if self.lane == "propose":
            props = self.view.event_proposals.get(seg.id)
            if props:
                p = props[0]
                note = (f" · ?{p.get('label')} {float(p['start_time']):.2f}"
                        f"–{float(p['end_time']):.2f}s")
        if seg.id in self.view.inserted_ids:
            # Synthetic chunk: its audio may exist ONLY in the original source
            # (inter-chunk gaps by construction) — decode a source slice, the
            # seam-audition path. Zero/near-zero width has nothing to sound.
            self.player.stop()
            self._stop_ticker()
            if seg.start_time is not None and seg.end_time is not None \
                    and float(seg.end_time) - float(seg.start_time) >= 0.02:
                self.run_worker(self._play_source_span(float(seg.start_time),
                                                       float(seg.end_time), note=note))
            return
        c = self.view.chunk(self.cursor)
        if c is None:
            self.player.stop()
            self._stop_ticker()
            return
        self.player.play(load_chunk(c.wav_path, c.start_s, c.end_s, speed=self.speed))
        if seg.start_time is not None and seg.end_time is not None:
            self._start_ticker(float(seg.start_time), float(seg.end_time), note)

    def _move(self, delta: int) -> None:
        if self.stage == "select":                 # the picker walks the source list
            if self._sources:
                self.cursor = max(0, min(len(self._sources) - 1, self.cursor + delta))
                self._render()
            return
        if self.stage == "spine":                  # the spine picker walks the skeletons
            if self._spines:
                self.cursor = max(0, min(len(self._spines) - 1, self.cursor + delta))
                self._render()
            return
        new = max(0, min(self.view.size - 1, self.cursor + delta))
        if abs(delta) == 1 and self._folded(new):
            # Single-step walking (keys + wheel) skips folded wordless inserts
            # (z, DEC fdb93036); explicit jumps (|delta| > 1) still land exactly.
            probe = new
            while 0 <= probe < self.view.size and self._folded(probe):
                probe += delta
            new = probe if 0 <= probe < self.view.size else self.cursor
        if new == self.cursor:
            return
        self.cursor = new
        self._word_cursor, self._word_anchor = 0, None  # word selection is per-segment
        now = time.monotonic()
        if now - self._state_saved > 1.0:   # bookmark survives crashes, not just quits
            save_tui_state(self._graph_db_path, self.view.source_id, new)
            self._state_saved = now
        self._render()
        if self.autoplay:
            self._play_cursor()

    def action_next(self) -> None:
        self._move(1)

    def action_prev(self) -> None:
        self._move(-1)

    def on_mouse_scroll_down(self, event) -> None:  # wheel = the same cursor move as keys
        self._move(1)

    def on_mouse_scroll_up(self, event) -> None:
        self._move(-1)

    def action_replay(self) -> None:
        self._play_cursor()

    async def action_seam_next(self) -> None:
        await self._audition_seam(1)

    async def action_seam_prev(self) -> None:
        await self._audition_seam(-1)

    async def _audition_seam(self, direction: int) -> None:
        """g/G: play the SOURCE audio across the boundary after/before the
        CURSOR SEGMENT — context tail + the whole gap + context head.

        Boundaries are fine-spine boundaries (first-drive correction,
        2026-07-23) — an FA cut, a real inter-chunk gap, or a coarse-segment
        crossing all audition the same way, because the decode goes back to the
        original source file, the only place the between-chunk audio exists
        (6beaa0e4, the de994164 missed-montage class). Everything that can fail
        is checked BEFORE any sound stops; Esc stops playback like any other
        chunk."""
        status = self.query_one("#status", Static)
        ref = self.view.seam(self.cursor, direction)
        if ref is None:
            status.update("seam audio: no neighbor segment in that direction")
            return
        path = self.view.source_path
        if not path or not Path(path).exists():
            status.update(f"seam audio: source media not found ({path or 'no path on Source'})")
            return
        self.player.stop()
        status.update(f"seam audio: decoding source {ref.start_s:.1f}–{ref.end_s:.1f}s …")
        try:
            samples = await load_source_slice(path, ref.start_s, ref.end_s,
                                              samplerate=self.player.samplerate)
        except (RuntimeError, OSError) as e:
            status.update(f"seam audio: decode failed — {e}")
            return
        if self.speed != 1.0 and len(samples):
            samples = stretch(samples, self.speed)
        self.player.play(samples)
        segs = self.view.segments
        status.update(
            f"♪ seam #{segs[ref.left].index}|#{segs[ref.right].index}:"
            f" source {ref.start_s:.1f}–{ref.end_s:.1f}s"
            f" (gap {ref.gap_s:+.2f}s) · esc stops")

    async def action_nudge_end_earlier(self) -> None:
        await self._nudge("end", -1)

    async def action_nudge_end_later(self) -> None:
        await self._nudge("end", 1)

    async def action_nudge_start_earlier(self) -> None:
        await self._nudge("start", -1)

    async def action_nudge_start_later(self) -> None:
        await self._nudge("start", 1)

    def action_nudge_step_down(self) -> None:
        self._step_nudge(-1)

    def action_nudge_step_up(self) -> None:
        self._step_nudge(1)

    def _step_nudge(self, delta: int) -> None:
        """{ / }: step the nudge increment along the ladder and persist it
        (sidecar preference, the speed pattern). First drive found the right
        granularity is per-BOUNDARY — 100ms fit some cuts, others needed
        20/10/5ms — so the step must adjust mid-walk, not per-launch."""
        cur = self._nudge_step * 1000.0
        i = min(range(len(self.NUDGE_STEPS_MS)),
                key=lambda j: abs(self.NUDGE_STEPS_MS[j] - cur))
        ms = self.NUDGE_STEPS_MS[max(0, min(len(self.NUDGE_STEPS_MS) - 1, i + delta))]
        self._nudge_step = ms / 1000.0
        save_tui_state(self._graph_db_path, self.view.source_id, self.cursor,
                       nudge_step_ms=ms)
        self.query_one("#status", Static).update(f"nudge step: {ms:g} ms")

    async def _nudge(self, edge: str, sign: int) -> None:
        """,/. (cursor END) and </> (cursor START): nudge a boundary TIME by
        ±--nudge-step-ms, then replay the updated cursor segment so the ear
        verifies at once (g/G stays the manual cross-boundary check).

        The 3f9948d6 surface over commit_time_nudge_correction: welded point
        cuts (sentence cuts share the exact boundary) move both edges in ONE
        atomic correction via plan_time_nudge; the journal records old/new per
        edge + the boundary words, so VAD+FA finetuning pairs derive straight
        from the correction journal (the flywheel). Key-repeat drops while a
        commit is in flight (the shift-throttle pattern); no review marker —
        a nudge is a time decision, not a text verdict."""
        if self._nudge_busy:
            return
        view, i = self.view, self.cursor
        status = self.query_one("#status", Static)
        delta = sign * self._nudge_step
        plan = plan_time_nudge(view.segments, i, edge, delta)
        if plan is None:
            status.update(f"nudge: refused ({edge} {delta:+.3f}s — missing times, "
                          "or a segment would collapse)")
            return
        segs = view.segments
        if edge == "end":
            left_t = segs[i].text
            right_t = segs[i + 1].text if i + 1 < view.size else ""
        else:
            left_t = segs[i - 1].text if i > 0 else ""
            right_t = segs[i].text
        words = {"left": (left_t.split() or [None])[-1],
                 "right": (right_t.split() or [None])[0]}
        self._nudge_busy = True
        try:
            await commit_time_nudge_correction(
                view.queue, view.graph_id, view.source_id, plan,
                self.session_id, boundary_words=words, step_s=delta,
                actor=self.actor, journal_path=self._journal_path)
        finally:
            self._nudge_busy = False
        by_id = {s.id: s for s in segs}
        for e in plan:   # local echo — the paint + replay read the nudged times
            s = by_id[e["segment_id"]]
            if e["edge"] == "start":
                s.start_time = e["new_time"]
            else:
                s.end_time = e["new_time"]
        self._render()
        # Immediate audible verification: replay the UPDATED CURSOR SEGMENT —
        # whether the word now fits its chunk is the thing the ear must judge
        # (user drive feedback: the g/G span muddied over/undershoot; press
        # g/G manually for cross-boundary context). END nudges replay only the
        # segment TAIL — a long segment must not make the ear wait to reach
        # the edge under judgment (second drive refinement).
        if segs[i].id in self.view.inserted_ids:
            self._play_cursor()   # synthetic chunk: the source-slice playback path
        elif edge == "end":
            c = self.view.chunk(i)
            if c is None:
                self.player.stop()
            else:
                tail = max(c.start_s, c.end_s - self.NUDGE_TAIL_S)
                self.player.play(load_chunk(c.wav_path, tail, c.end_s, speed=self.speed))
        else:
            self._play_cursor()
        e0 = plan[0]
        welded = " ⚭" if len(plan) > 1 else ""
        status.update(
            f"⏱ #{segs[i].index} {e0['edge']} {e0['old_time']:.2f}→{e0['new_time']:.2f}s"
            f" ({delta:+.3f}s){welded} · replaying segment")

    def _step_speed(self, delta: int) -> None:
        """Step the playback rate along the preset ladder, re-sound the chunk at the
        new rate (immediate audible confirmation), persist the preference (sidecar —
        view state like the cursor bookmark, never a graph write)."""
        i = min(range(len(self.SPEEDS)), key=lambda j: abs(self.SPEEDS[j] - self.speed))
        self.speed = self.SPEEDS[max(0, min(len(self.SPEEDS) - 1, i + delta))]
        save_tui_state(self._graph_db_path, self.view.source_id, self.cursor,
                       speed=self.speed)
        self._render()
        self._play_cursor()

    def action_speed_down(self) -> None:
        self._step_speed(-1)

    def action_speed_up(self) -> None:
        self._step_speed(1)

    def action_yank(self) -> None:
        """Copy the focused segment's effective text to the system clipboard —
        sharing a segment must not require a screenshot or re-typing.

        A clipboard TOOL (wl-copy/xclip/xsel) is the primary path: OSC 52 is
        fire-and-forget and VTE terminals commonly reject it (drive round 5),
        so it stays only as the fallback for tool-less hosts."""
        seg = self.view.segments[self.cursor]
        via = self._copy_system(seg.text)
        if via is None:
            self.copy_to_clipboard(seg.text)   # OSC 52 — may be ignored by the terminal
            via = "osc52, terminal-dependent"
        self.query_one("#status", Static).update(
            f"copied segment #{seg.index} text ({len(seg.text)} chars, {via})")

    def _copy_system(self, text: str) -> Optional[str]:
        """Pipe text to the first available system clipboard tool; None = no tool took it."""
        for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"],
                    ["xsel", "--clipboard", "--input"]):
            if shutil.which(cmd[0]) is None:
                continue
            try:
                subprocess.run(cmd, input=text.encode(), check=True, timeout=2)
                return cmd[0]
            except (OSError, subprocess.SubprocessError):
                continue
        return None

    def action_edit(self) -> None:
        editor = self.query_one("#editor", Input)
        self._input_mode = "edit"
        editor.value = self.view.segments[self.cursor].text
        editor.display = True
        editor.focus()

    async def on_input_submitted(self, event) -> None:
        if self._input_mode == "mark":
            await self._submit_mark(event.value)
            return
        if self._input_mode == "insert":
            await self._submit_insert(event.value)
            return
        if self._input_mode == "assign":
            await self._submit_assign(event.value)
            return
        if self._input_mode == "split":
            await self._submit_split(event.value, event.input.cursor_position)
            return
        if self._input_mode == "propose_split":
            await self._submit_propose_split(event.value, event.input.cursor_position)
            return
        if self._input_mode == "relabel":
            await self._submit_relabel(event.value)
            return
        if self._input_mode == "gate":
            await self._submit_gate(event.value)
            return
        if self._input_mode == "annotate":
            await self._submit_annotate(event.value)
            return
        seg = self.view.segments[self.cursor]
        new_text = event.value
        if new_text != seg.text:
            await commit_text_correction(
                self.view.queue, self.view.graph_id, self.view.source_id,
                seg.id, new_text, self.session_id,
                old_text=seg.text, actor=self.actor,
                journal_path=self._journal_path)
            seg.text = new_text          # local echo of the new effective text
            self._marks[self.cursor] = "corrected"
            # Text arriving/leaving changes proposal eligibility (empty chunks
            # never propose): a missed-speech insert gains its cluster chip the
            # moment its words land, an emptied chunk drops it.
            self.view.refresh_turn_proposal(seg.id)
        # A text-bearing PRUNED position must leave the prune set (the same
        # rescue as boundary shifts): the prune otherwise drops the position —
        # WITH its restored text — from the downstream effective view. Fires
        # on re-submit too (recovery path for edits made before this guard).
        if new_text.strip() and seg.id in self.view.pruned_ids:
            prior = self.view.prune_correction_for(seg.id)
            if prior is not None:
                amended = await commit_prune_amendment(
                    self.view.queue, self.view.graph_id, prior, [seg.id],
                    self.session_id, actor=self.actor,
                    journal_path=self._journal_path)
                self.view.unprune_local(prior["id"], amended)
                self._marks[self.cursor] = "corrected"
        self._close_editor()
        self._render()

    def _close_editor(self) -> None:
        editor = self.query_one("#editor", Input)
        editor.display = False
        self.set_focus(None)
        self._input_mode = "edit"

    def action_cycle_lane(self) -> None:
        """tab: cycle the pass lane (walk <-> assign). Lane is a VIEW preference
        (sidecar-persisted, db-wide) — corrections are spine state, the lane
        only scopes which vocabulary is live (DEC cc55a7b5 / 8a4df244).

        priority=True on the binding: Textual's Screen binds tab to focus_next,
        which silently shadows app-level BINDINGS (first-drive find 2026-07-25
        — tab did nothing); the escape binding is the in-repo precedent. The
        editor guard keeps a priority tab from hijacking an open edit."""
        if self.query_one("#editor", Input).display:
            return
        self._cycle_lane(1)

    def action_cycle_lane_prev(self) -> None:
        """shift+tab: cycle the pass lane BACKWARD (drive ask 2026-08-03 — the
        walk<->annotate two-step of restore-then-annotate sits at opposite
        ends of the forward rotation). Same gate, editor guard, and sidecar
        persistence as tab; priority=True for the same Screen-shadowing
        reason (Screen binds shift+tab to focus_previous)."""
        if self.query_one("#editor", Input).display:
            return
        self._cycle_lane(-1)

    def _cycle_lane(self, delta: int) -> None:
        # walk -> assign -> [propose ->] annotate -> walk; the propose lane only
        # enters the rotation when a proposal set loaded (no set = the walk
        # stays manual); the annotate lane is always available (fc42614d).
        order = (["walk", "assign"] + (["propose"] if self.view.proposals_meta else [])
                 + ["annotate"])
        self.lane = order[(order.index(self.lane) + delta) % len(order)] \
            if self.lane in order else "walk"
        self._word_anchor = None
        save_tui_state(self._graph_db_path, self.view.source_id, None, lane=self.lane)
        self._render()
        if self.lane == "annotate":
            menu = self._overlay_label_menu()
            self.query_one("#status", Static).update(
                "annotate: h/l walk words · v range · space commits ◈"
                + self._overlay_label + " · "
                + " ".join(f"{i + 1}:{c}" for i, c in enumerate(menu[:6]))
                + (" · …" if len(menu) > 6 else "") + " · A other")
        if self.lane == "assign":
            menu = self._assign_menu()
            if menu:
                self.query_one("#status", Static).update(
                    "assign: " + " · ".join(f"{i + 1}:{nm}" for i, (_, nm) in enumerate(menu[:6]))
                    + (" · …" if len(menu) > 6 else "") + " · A new")

    def _entity_name(self, entity_id: Optional[str]) -> str:
        """Display name for an entity id; provisional handles read with a
        leading ? (a DESCRIPTION, not an identification — DEC 484e2d74)."""
        for d in self._entities:
            if d.get("id") == entity_id:
                p = d.get("properties") or {}
                nm = str(p.get("canonical_name") or str(entity_id)[:8])
                return f"?{nm}" if p.get("provisional") else nm
        return str(entity_id or "")[:8]

    _CLUSTER_TINTS = ("cyan", "green", "yellow", "blue", "bright_magenta", "bright_red")

    def _cluster_style(self, cluster: str) -> str:
        """Stable per-cluster tint (dim — a proposal reads quieter than an
        assignment's magenta): index by sorted cluster label within this
        source's proposals; labels are result-scoped so stability only needs
        to hold per open."""
        clusters = sorted({str(p.get("cluster")) for p in self.view.turn_proposals.values()})
        idx = clusters.index(cluster) if cluster in clusters else 0
        return f"dim {self._CLUSTER_TINTS[idx % len(self._CLUSTER_TINTS)]}"

    def _assign_menu(self) -> List[Tuple[str, str]]:
        """The layered digit menu (DEC 4ec6a49c): THIS source's assigned
        speakers first (spine encounter order), then the rest of the registry
        (name order) — capped at the 9 digit keys; A mints new."""
        seen: List[str] = []
        for s in self.view.segments:
            sp = self.view.speakers.get(s.id)
            if sp and sp.get("entity_id") and sp["entity_id"] not in seen:
                seen.append(sp["entity_id"])
        rest = [d["id"] for d in self._entities if d["id"] not in seen]
        return [(eid, self._entity_name(eid)) for eid in (seen + rest)[:9]]

    async def action_assign_pick(self, n: int) -> None:
        """1-9: assign the cursor segment to menu speaker #n (and make it the
        active speaker the space run continues)."""
        menu = self._assign_menu()
        if not (1 <= n <= len(menu)):
            self.query_one("#status", Static).update(
                f"assign: no speaker #{n} — A mints a new one")
            return
        self._active_entity = menu[n - 1][0]
        await self._commit_assign(menu[n - 1][0])

    async def action_assign_same(self) -> None:
        """space (assign lane): same speaker continues — assign the ACTIVE
        entity to the cursor segment and advance. One keystroke per segment:
        the single-narrator fast path (DEC 8a4df244)."""
        if self._active_entity is None:
            self.query_one("#status", Static).update(
                "assign: no active speaker — pick 1-9 or A new")
            return
        await self._commit_assign(self._active_entity)

    async def action_assign_accept(self) -> None:
        """a (assign lane): ACCEPT the cursor segment's proposed cluster — the
        BULK cluster-name-once op (DEC 8a4df244): one keystroke assigns the
        bound entity to EVERY unassigned segment this cluster dominates; the
        digit/space vocabulary stays the per-segment correction layer over it.
        An unbound cluster hops through the speaker editor first (digit pick /
        Name / ?handle — the A vocabulary); accepting a second cluster onto an
        already-bound entity records verdict=cluster-merge (embeddings split
        one physical voice — prime flywheel supervision, DEC d6df3a8e)."""
        seg = self.view.segments[self.cursor]
        prop = self.view.turn_proposals.get(seg.id)
        if not prop:
            self.query_one("#status", Static).update(
                "accept: no diarization proposal on this segment")
            return
        cluster = str(prop["cluster"])
        entity = self.view.cluster_entities.get(cluster)
        if entity:
            await self._commit_accept(cluster, entity)
            return
        self._accept_cluster = cluster
        editor = self.query_one("#editor", Input)
        self._input_mode = "assign"
        editor.value = ""
        editor.display = True
        editor.focus()
        menu = self._assign_menu()
        listing = " · ".join(f"{i + 1}:{nm}" for i, (_, nm) in enumerate(menu))
        self.query_one("#status", Static).update(
            f'accept {cluster}: #-or-Name · "? handle" = provisional'
            + (f" · {listing}" if listing else ""))

    async def _commit_accept(self, cluster: str, entity_id: str) -> None:
        """One bulk speaker_assign over every UNASSIGNED segment the cluster
        dominates — assigned segments keep their per-segment judgments (the
        latest-wins correction layer is never clobbered by a later accept).
        Verdict: accept for a fresh cluster binding, cluster-merge when the
        entity already carries a DIFFERENT cluster."""
        targets = [s.id for s in self.view.segments
                   if s.id not in self.view.speakers
                   and str((self.view.turn_proposals.get(s.id) or {}).get("cluster")) == cluster]
        if not targets:
            self.query_one("#status", Static).update(
                f"accept: no unassigned segments under {cluster}")
            return
        merged = entity_id in {e for c, e in self.view.cluster_entities.items()
                               if c != cluster}
        verdict = "cluster-merge" if merged else "accept"
        cov = [float((self.view.turn_proposals.get(t) or {}).get("coverage") or 0.0)
               for t in targets]
        cap = self.view.turns_meta.get("capability") or {}
        meta = self.view.turns_meta.get("metadata") or {}
        proposal = {"cluster": cluster,
                    "model_id": meta.get("model_id"),
                    "config_hash": cap.get("config_hash"),
                    "segments": len(targets),
                    "mean_coverage": round(sum(cov) / len(cov), 3) if cov else None}
        corr_id = await commit_speaker_assign_correction(
            self.view.queue, self.view.graph_id, self.view.source_id,
            targets, entity_id, self.session_id, verdict=verdict,
            proposal=proposal, actor=self.actor, journal_path=self._journal_path)
        self.view.assign_local(targets, entity_id, verdict, corr_id, cluster=cluster)
        self._active_entity = entity_id
        self._render()
        self.query_one("#status", Static).update(
            f"{verdict}: {cluster} → {self._entity_name(entity_id)}"
            f" ({len(targets)} segments)")

    def action_assign_new(self) -> None:
        """A: the speaker editor — a bare digit picks from the numbered menu
        (the M mark-editor pattern; scales past the 1-9 direct keys), `Name`
        mints, `? descriptive handle` mints PROVISIONAL (distinct voice,
        unknown identity; DEC 484e2d74). Exact name matches reuse the existing
        entity instead of duplicating."""
        editor = self.query_one("#editor", Input)
        self._input_mode = "assign"
        editor.value = ""
        editor.display = True
        editor.focus()
        menu = self._assign_menu()
        listing = " · ".join(f"{i + 1}:{nm}" for i, (_, nm) in enumerate(menu))
        self.query_one("#status", Static).update(
            'speaker: #-or-Name · "? handle" = provisional'
            + (f" · {listing}" if listing else ""))

    async def _submit_assign(self, raw: str) -> None:
        self._close_editor()
        token = (raw or "").strip()
        if token.isdigit():
            # Bare digit = menu pick (the mark-editor precedent) — the path
            # that scales when a source carries more speakers than digit keys.
            menu = self._assign_menu()
            n = int(token)
            if not (1 <= n <= len(menu)):
                self._render()
                self.query_one("#status", Static).update(
                    f"assign: no speaker #{n} — menu has {len(menu)}")
                return
            self._active_entity = menu[n - 1][0]
            await self._commit_assign(menu[n - 1][0])
            return
        parsed = parse_entity_input(raw)
        if parsed is None:
            self._render()
            return
        name, provisional = parsed
        for d in self._entities:   # exact-name reuse: the registry stays deduplicated
            p = d.get("properties") or {}
            if str(p.get("canonical_name") or "").lower() == name.lower():
                self._active_entity = d["id"]
                await self._commit_assign(d["id"])
                return
        eid = await commit_speaker_entity(
            self.view.queue, self.view.graph_id, name, self.session_id,
            provisional=provisional, actor=self.actor,
            journal_path=self._journal_path)
        self._entities.append({"id": eid, "properties": {
            "canonical_name": name, "provisional": provisional, "kind": "person"}})
        self._active_entity = eid
        await self._commit_assign(eid)

    async def _commit_assign(self, entity_id: str) -> None:
        """Commit one speaker assignment on the cursor segment and advance.

        verdict=name (the proposal-less manual walk — accept/cluster-merge
        activate when diarization proposals exist, DEC 8a4df244); reassignment
        needs no supersede: the projection is latest-wins per segment, the
        re-decision CHAIN is the record (the nudge precedent)."""
        if self._accept_cluster is not None:
            # The a-gesture's editor hop lands here: the picked/minted entity
            # names the PENDING CLUSTER (bulk), not just the cursor segment.
            cluster, self._accept_cluster = self._accept_cluster, None
            await self._commit_accept(cluster, entity_id)
            return
        seg = self.view.segments[self.cursor]
        corr_id = await commit_speaker_assign_correction(
            self.view.queue, self.view.graph_id, self.view.source_id,
            [seg.id], entity_id, self.session_id, verdict="name",
            actor=self.actor, journal_path=self._journal_path)
        self.view.assign_local([seg.id], entity_id, "name", corr_id)
        idx = seg.index
        self._move(1)
        self._render()
        self.query_one("#status", Static).update(
            f"@ #{idx} → {self._entity_name(entity_id)}")

    def action_toggle_tier2(self) -> None:
        """t (propose lane): show/hide the audition tier (3a5cb858 shape A).

        Tier-2 spans are the model's below-threshold catches — audition-only,
        never carve cuts. Hidden by default so the primary walk stays the
        operating-point contract; shown, they join the pending walk and ride
        the SAME accept machinery (an accept is bench data at its tier)."""
        view = self.view
        status = self.query_one("#status", Static)
        if not (view.proposals_meta or {}).get("tier2_total"):
            status.update("single-tier proposal set — no audition tier to show")
            return
        view.show_tier2 = not view.show_tier2
        view.refresh_event_proposals()
        self._render()
        t2 = (view.proposals_meta or {}).get("tier2_total", 0)
        status.update(f"audition tier shown ({t2} tier-2 spans join the walk) · t hides"
                      if view.show_tier2 else "audition tier hidden · t shows")

    async def action_propose_accept(self) -> None:
        """a (propose lane): accept the cursor anchor's first pending proposal.

        The accept gesture IS the insert op (DEC 8e05b87b) — HOW it lands
        depends on where the span sits (drive find 2026-07-29: a mid-chunk
        accept that only grafts an overlapping insert leaves the original
        chunk still covering the breath):
          - in the GAP after the anchor: plain labeled insert;
          - STRADDLING a chunk edge: insert + boundary pull(s) — the flanking
            chunk edges nudge clear of the span (nudges = the edit record);
          - strictly INSIDE a splittable anchor: the SPLIT CHAIN — an editor
            hop places the text cut (caret pre-seeded proportionally to the
            span start), then split + insert-between + right-half pull land
            as three journaled ops (x unsplits / x removes undo them);
          - inside but NOT splittable (synthetic or <2 words): overlay insert
            only, S splits manually — never a silent coverage loss."""
        view = self.view
        i = self.cursor
        seg = view.segments[i]
        status = self.query_one("#status", Static)
        props = view.event_proposals.get(seg.id)
        if not props:
            status.update("no pending proposal at cursor — n/N jump to one")
            return
        p = props[0]
        ps, pe = float(p["start_time"]), float(p["end_time"])
        eps = 0.05
        a_start = float(seg.start_time) if seg.start_time is not None else None
        a_end = float(seg.end_time) if seg.end_time is not None else None
        interior = (a_start is not None and a_end is not None
                    and ps > a_start + eps and pe < a_end - eps)
        # plan_chunk_split handles synthetics uniformly (anchors resolve past
        # them) — only textless/one-word chunks can't divide.
        splittable = interior and len((seg.text or "").split()) >= 2
        if splittable:
            # The editor hop (the assign a-gesture precedent): the human owns
            # the TEXT division, the model owns the cut TIME.
            self._pending_proposal = (i, p)
            editor = self.query_one("#editor", Input)
            self._input_mode = "propose_split"
            editor.value = seg.text
            editor.display = True
            editor.focus()
            frac = (ps - a_start) / max(a_end - a_start, 1e-6)
            editor.cursor_position = max(0, min(len(seg.text),
                                                round(len(seg.text) * frac)))
            status.update(f"accept: caret marks the text cut at {ps:.2f}s"
                          " · enter = split + inhale between · esc cancels")
            return

        plan = plan_chunk_insert(view.segments, i, inserted_ids=view.inserted_ids,
                                 insert_ranks=view.insert_ranks)
        if plan is None:
            status.update("accept: refused (missing times, or an overlapping "
                          "boundary — nudge the overlap first)")
            return
        nxt = view.segments[i + 1] if i + 1 < view.size else None
        insert_id = await commit_chunk_insert_correction(
            view.queue, view.graph_id, view.source_id,
            plan["after_id"], ps, pe, self.session_id,
            before_segment_id=plan["before_id"], label=p.get("label"),
            rank=plan["rank"], actor=self.actor, journal_path=self._journal_path)
        pos = view.add_insert_local(
            {"id": insert_id,
             "payload": {"operation": "chunk_insert",
                         "after_segment_id": plan["after_id"],
                         "start_time": ps, "end_time": pe,
                         "label": p.get("label"), "text": "", "rank": plan["rank"]}})
        note = ""
        if not interior:
            # Straddle pulls: a span reaching into a flanking chunk pulls that
            # edge clear (the accepted span owns its time; the pull is spine
            # truth the bench reads as part of the accept).
            if a_end is not None and ps < a_end - eps:
                last = ((seg.text or "").split() or [None])[-1]
                await self._commit_span_nudge(seg.id, "end", a_end, ps,
                                              {"left": last, "right": None})
                note = " · anchor end pulled"
            if nxt is not None and nxt.start_time is not None \
                    and pe > float(nxt.start_time) + eps:
                first = ((nxt.text or "").split() or [None])[0]
                await self._commit_span_nudge(nxt.id, "start",
                                              float(nxt.start_time), pe,
                                              {"left": None, "right": first})
                note = note or " · next start pulled"
        else:
            note = " · mid-chunk, text not divisible — overlay insert only"
        if pos is not None:
            self._marks = {(k + 1 if k >= pos else k): v for k, v in self._marks.items()}
            self.cursor = pos
        view.refresh_event_proposals()  # the accepted span now occupies — pending re-derives
        self._render()
        self.player.stop()
        self.run_worker(self._play_source_span(
            ps, pe, note=f" · ✓ {p.get('label')} accepted{note}"))

    async def _submit_propose_split(self, value: str, caret: int) -> None:
        """The propose-accept editor hop's submission: split the anchor at the
        PROPOSAL's start (the caret only divides text — the model owns the
        seed time, unlike S where the caret interpolates it), insert the
        labeled span between the halves, and pull the right half's start past
        the span end. One gesture, three journaled ops."""
        self._close_editor()
        view = self.view
        status = self.query_one("#status", Static)
        pending, self._pending_proposal = self._pending_proposal, None
        if pending is None:
            self._render()
            return
        i, p = pending
        seg = view.segments[i] if 0 <= i < view.size else None
        head = (view.event_proposals.get(seg.id) or [None])[0] if seg else None
        if head is None or head.get("proposal_id") != p.get("proposal_id"):
            self._render()
            status.update("accept: proposal state changed — n/N jump again")
            return
        ps, pe = float(p["start_time"]), float(p["end_time"])
        plan = plan_chunk_split(view.segments, i, caret, text=value,
                                inserted_ids=view.inserted_ids)
        if plan is None:
            # Caret at an extreme is the BOOKEND signal (drive find 2026-07-31,
            # source-1 pass 2): an interior-classified span that in fact hugs
            # a chunk edge (stopped short of it beyond eps — FA edge drift)
            # has no dividable text on one side. All words LEFT of the caret =
            # end-bookend; all RIGHT = start-bookend. Resolve as the straddle
            # shape does — pull the edge clear, land the insert; no split.
            left_words = value[:caret].split()
            right_words = value[caret:].split()
            if left_words and not right_words:
                await self._accept_bookend(i, p, "end")
                return
            if right_words and not left_words:
                await self._accept_bookend(i, p, "start")
                return
            self._render()
            status.update("accept: split refused (the caret must leave words "
                          "on both sides of the cut)")
            return
        plan["split_s"] = ps   # the MODEL's span start is the cut, not the caret fraction
        old_text = seg.text
        ids = await commit_chunk_split_correction(
            view.queue, view.graph_id, view.source_id, plan["segment_id"],
            plan["split_s"], plan["left_text"], plan["right_text"], plan["end_s"],
            self.session_id, plan["after_id"], before_segment_id=plan["before_id"],
            old_text=old_text, boundary_words=plan["boundary_words"],
            actor=self.actor, journal_path=self._journal_path)
        pos_r = view.split_local(i, plan["left_text"], plan["split_s"],
                                 {"id": ids["insert_id"],
                                  "payload": {"operation": "chunk_insert",
                                              "after_segment_id": plan["after_id"],
                                              "start_time": plan["split_s"],
                                              "end_time": plan["end_s"],
                                              "label": None,
                                              "text": plan["right_text"]}})
        view.split_groups[ids["insert_id"]] = {
            "group_ids": [ids["text_id"], ids["nudge_id"]],
            "target_id": plan["segment_id"], "old_text": old_text,
            "old_end": plan["end_s"]}
        if pos_r is not None:
            self._marks = {(k + 1 if k >= pos_r else k): v for k, v in self._marks.items()}
        iplan = plan_chunk_insert(view.segments, i, inserted_ids=view.inserted_ids,
                                  insert_ranks=view.insert_ranks)
        if iplan is None:
            self._render()
            status.update("accept: split landed but the between-insert "
                          "refused — i inserts manually")
            return
        insert_id = await commit_chunk_insert_correction(
            view.queue, view.graph_id, view.source_id,
            iplan["after_id"], ps, pe, self.session_id,
            before_segment_id=iplan["before_id"], label=p.get("label"),
            rank=iplan["rank"], actor=self.actor, journal_path=self._journal_path)
        pos = view.add_insert_local(
            {"id": insert_id,
             "payload": {"operation": "chunk_insert",
                         "after_segment_id": iplan["after_id"],
                         "start_time": ps, "end_time": pe,
                         "label": p.get("label"), "text": "", "rank": iplan["rank"]}})
        # the split welded the right half at ps; the span owns [ps, pe] now
        right_first = ((plan["right_text"] or "").split() or [None])[0]
        await self._commit_span_nudge(ids["insert_id"], "start", ps, pe,
                                      {"left": None, "right": right_first})
        if pos is not None:
            self._marks = {(k + 1 if k >= pos else k): v for k, v in self._marks.items()}
            self.cursor = pos
        view.refresh_event_proposals()  # the accepted span now occupies — pending re-derives
        self._render()
        self.player.stop()
        self.run_worker(self._play_source_span(
            ps, pe, note=f" · ✓ {p.get('label')} isolated (split + insert)"))

    async def _accept_bookend(self, i: int, p: Dict[str, Any], edge: str) -> None:
        """Resolve a caret-at-extreme propose-accept as a BOOKEND accept
        (drive find 2026-07-31): the span sits strictly inside the anchor's
        recorded time yet the driver marked every word on one side — the
        event hugs that edge in fact, the chunk's recorded edge just drifted
        past it. Same three-part shape as the straddle accept: pull the edge
        clear of the span (the pull is the edit record the bench reads),
        land the labeled insert in the seam, replay the span."""
        view = self.view
        status = self.query_one("#status", Static)
        seg = view.segments[i]
        ps, pe = float(p["start_time"]), float(p["end_time"])
        seam = i if edge == "end" else i - 1
        if seam < 0:
            self._render()
            status.update("accept: no seam before the first segment — "
                          "i inserts manually")
            return
        plan = plan_chunk_insert(view.segments, seam, inserted_ids=view.inserted_ids,
                                 insert_ranks=view.insert_ranks)
        if plan is None:
            self._render()
            status.update("accept: refused (missing times, or an overlapping "
                          "boundary — nudge the overlap first)")
            return
        words = (seg.text or "").split() or [None]
        if edge == "end":
            await self._commit_span_nudge(seg.id, "end", float(seg.end_time), ps,
                                          {"left": words[-1], "right": None})
        else:
            await self._commit_span_nudge(seg.id, "start", float(seg.start_time), pe,
                                          {"left": None, "right": words[0]})
        insert_id = await commit_chunk_insert_correction(
            view.queue, view.graph_id, view.source_id,
            plan["after_id"], ps, pe, self.session_id,
            before_segment_id=plan["before_id"], label=p.get("label"),
            rank=plan["rank"], actor=self.actor, journal_path=self._journal_path)
        pos = view.add_insert_local(
            {"id": insert_id,
             "payload": {"operation": "chunk_insert",
                         "after_segment_id": plan["after_id"],
                         "start_time": ps, "end_time": pe,
                         "label": p.get("label"), "text": "", "rank": plan["rank"]}})
        if pos is not None:
            self._marks = {(k + 1 if k >= pos else k): v for k, v in self._marks.items()}
            self.cursor = pos
        view.refresh_event_proposals()  # the accepted span now occupies — pending re-derives
        self._render()
        self.player.stop()
        self.run_worker(self._play_source_span(
            ps, pe, note=f" · ✓ {p.get('label')} accepted · anchor {edge} pulled"))

    async def _commit_span_nudge(self, segment_id: str, edge: str, old_t: float,
                                 new_t: float, words: Dict[str, Any]) -> None:
        """One ABSOLUTE boundary move as a time-nudge correction (manual plan —
        the accept chain knows target times, not ladder deltas) + local echo."""
        plan = [{"segment_id": segment_id, "edge": edge,
                 "old_time": old_t, "new_time": new_t}]
        await commit_time_nudge_correction(
            self.view.queue, self.view.graph_id, self.view.source_id, plan,
            self.session_id, boundary_words=words, step_s=new_t - old_t,
            actor=self.actor, journal_path=self._journal_path)
        seg = next((s for s in self.view.segments if s.id == segment_id), None)
        if seg is not None:
            if edge == "start":
                seg.start_time = new_t
            else:
                seg.end_time = new_t

    def action_propose_audition(self) -> None:
        """R (propose lane): audition the cursor anchor's first pending
        proposal span — pairs with r (chunk replay) so actual and proposed
        line up by ear (drive ask 2026-07-29); the live ticker shows position."""
        props = self.view.event_proposals.get(self.view.segments[self.cursor].id)
        if not props:
            self.query_one("#status", Static).update(
                "no pending proposal at cursor — n/N jump to one")
            return
        p = props[0]
        self.player.stop()
        self.run_worker(self._play_source_span(
            float(p["start_time"]), float(p["end_time"]),
            note=f" · ?{p.get('label')} score {float(p.get('score') or 0):.2f} · a accepts"))

    def action_propose_next(self) -> None:
        """n (propose lane): jump to the next pending proposal and audition it."""
        self._jump_proposal(+1)

    def action_propose_prev(self) -> None:
        """N (propose lane): jump to the previous pending proposal and audition it."""
        self._jump_proposal(-1)

    def _jump_proposal(self, direction: int) -> None:
        """Cursor to the nearest anchor with pending proposals in `direction`
        and AUDITION the first one's source span — judge on one keypress,
        accept on the next (the bench pass's assist rhythm)."""
        view = self.view
        rng = (range(self.cursor + 1, view.size) if direction > 0
               else range(self.cursor - 1, -1, -1))
        for i in rng:
            props = view.event_proposals.get(view.segments[i].id)
            if props:
                self.cursor = i
                self._render()
                p = props[0]
                self.player.stop()
                self.run_worker(self._play_source_span(
                    float(p["start_time"]), float(p["end_time"]),
                    note=f" · ?{p.get('label')} score {float(p.get('score') or 0):.2f}"
                         " · a accepts · R replays"))
                return
        self.query_one("#status", Static).update("no more pending proposals this way")

    async def action_insert_chunk(self) -> None:
        """i: insert a chunk into the gap AFTER the cursor (DEC 3d3fa2a8) —
        whole-gap span, one keystroke (the de994164 missed-dispatch case);
        ZERO-WIDTH at a welded cut, grown over the bookends by the nudge keys
        (insert+nudge completes non-speech isolation with no new machinery)."""
        await self._insert_chunk(None)

    def _insert_label_menu(self) -> List[str]:
        """Selectable insert labels: the recommended slate first, then labels
        carried by this source's ACTIVE inserts (the mark-class menu pattern —
        open vocabulary, slate is DATA; a proven label persists by promotion
        into RECOMMENDED_INSERT_LABELS)."""
        return list(RECOMMENDED_INSERT_LABELS) + [
            c for c in self.view.seen_insert_labels
            if c not in RECOMMENDED_INSERT_LABELS]

    def action_insert_labeled(self) -> None:
        """I: labeled insert — the annotation-class editor (open vocabulary,
        pre-filled with the last-used label; a leading digit picks from the
        numbered menu): inhale/hesitation bookends become LABELED spans, the
        VAD-gold flywheel record."""
        if self._plan_insert() is None:
            return
        editor = self.query_one("#editor", Input)
        self._input_mode = "insert"
        editor.value = f"{self._insert_label} "
        editor.display = True
        editor.focus()
        menu = self._insert_label_menu()
        self.query_one("#status", Static).update(
            "insert label: class-or-# · "
            + " ".join(f"{i + 1}:{c}" for i, c in enumerate(menu)))

    def _plan_insert(self) -> Optional[Dict[str, Any]]:
        """Plan the insert after the cursor; refusals paint status (None).
        Synthetic neighbors are fine — anchors resolve past them, so sibling
        inserts stack in one gap (inhale · um · inhale, the C.1 drive find)."""
        view, i = self.view, self.cursor
        plan = plan_chunk_insert(view.segments, i, inserted_ids=view.inserted_ids,
                                 insert_ranks=view.insert_ranks)
        if plan is None:
            self.query_one("#status", Static).update(
                "insert: refused (missing times, or an overlapping "
                "boundary — nudge the overlap first)")
            return None
        return plan

    async def _insert_chunk(self, label: Optional[str]) -> None:
        """Commit one chunk insertion + local echo; the cursor lands ON the new
        chunk so nudges/edit/label apply at once."""
        plan = self._plan_insert()
        if plan is None:
            self._render()
            return
        view = self.view
        insert_id = await commit_chunk_insert_correction(
            view.queue, view.graph_id, view.source_id,
            plan["after_id"], plan["start_s"], plan["end_s"], self.session_id,
            before_segment_id=plan["before_id"], label=label,
            rank=plan["rank"], actor=self.actor, journal_path=self._journal_path)
        pos = view.add_insert_local(
            {"id": insert_id,
             "payload": {"operation": "chunk_insert",
                         "after_segment_id": plan["after_id"],
                         "start_time": plan["start_s"], "end_time": plan["end_s"],
                         "label": label, "text": "", "rank": plan["rank"]}})
        if pos is not None:
            # _marks is POSITIONAL (cursor -> decision echo): positions at/after
            # the splice shift right — the walk-indexing perturbation the design
            # priced in (DEC 3d3fa2a8 known cost).
            self._marks = {(k + 1 if k >= pos else k): v for k, v in self._marks.items()}
            self.cursor = pos
        self._render()
        lab = f" [{label}]" if label else ""
        status = self.query_one("#status", Static)
        if plan["welded"]:
            status.update(f"⊕ zero-width insert{lab} at {plan['start_s']:.2f}s"
                          " — grow it with ,/. </>")
        else:
            status.update(f"⊕ inserted{lab} {plan['start_s']:.2f}–{plan['end_s']:.2f}s"
                          " · playing source · e types its text")
            self.player.stop()
            self.run_worker(self._play_source_span(plan["start_s"], plan["end_s"]))

    async def _submit_insert(self, raw: str) -> None:
        """The I-editor submission: first token = the annotation class; a
        leading digit resolves against the numbered menu (the M-picker
        grammar, shared resolver)."""
        self._close_editor()
        raw, err = resolve_mark_class_token(raw, self._insert_label_menu())
        if err:
            self._render()
            self.query_one("#status", Static).update(f"insert: {err}")
            return
        tokens = (raw or "").split()
        if not tokens:
            self._render()
            return
        label = tokens[0].strip('`"\'')
        if not label or not label[:1].isalnum():
            self._render()
            self.query_one("#status", Static).update(
                "insert: label must start with a letter or digit")
            return
        self._insert_label = label
        save_tui_state(self._graph_db_path, self.view.source_id, self.cursor,
                       insert_label=label)
        await self._insert_chunk(label)

    def action_split_chunk(self) -> None:
        """S: split the cursor chunk at a word boundary (work item 99c1d2ba) —
        the dual of i-insert: a boundary INSIDE the chunk. The editor opens
        with the segment's text; place the caret where the cut belongs and
        enter commits — the seed time interpolates the caret's character
        fraction, and the { } ladder + ,/. nudges + g audition own the
        precision (sub-word truth comes from nudging AFTER the split). Unlocks
        the flywheel spans locked inside original VAD chunks: split before and
        after an inhale/um, then mark or label the isolated middle."""
        seg = self.view.segments[self.cursor]
        status = self.query_one("#status", Static)
        if seg.id in self.view.pruned_ids:
            status.update("split: pruned position — e-edit text first (rescue), then split")
            return
        if len((seg.text or "").split()) < 2:
            status.update("split: needs at least two words (both halves must keep text)")
            return
        editor = self.query_one("#editor", Input)
        self._input_mode = "split"
        editor.value = seg.text
        editor.display = True
        editor.focus()
        editor.cursor_position = 0
        status.update("split: place the caret at the cut point · enter splits · esc cancels")

    async def _submit_split(self, value: str, caret: int) -> None:
        """The S-editor submission: split the cursor chunk at the caret (the
        text AS SUBMITTED partitions, so a typo fixed while placing the caret
        rides the halves). Commit = ONE atomic batch of the three composed
        verbs + ONE chunk-split journal op (the new-boundary flywheel record);
        the local echo mirrors the projection and the cursor stays on the LEFT
        half so ,/. tunes the new welded seam at once."""
        self._close_editor()
        view, i = self.view, self.cursor
        seg = view.segments[i]
        status = self.query_one("#status", Static)
        plan = plan_chunk_split(view.segments, i, caret, text=value,
                                inserted_ids=view.inserted_ids)
        if plan is None:
            self._render()
            status.update("split: refused (the caret must leave words on both"
                          " sides of the cut, and the chunk needs audio times)")
            return
        old_text = seg.text
        ids = await commit_chunk_split_correction(
            view.queue, view.graph_id, view.source_id, plan["segment_id"],
            plan["split_s"], plan["left_text"], plan["right_text"], plan["end_s"],
            self.session_id, plan["after_id"], before_segment_id=plan["before_id"],
            old_text=old_text, boundary_words=plan["boundary_words"],
            actor=self.actor, journal_path=self._journal_path)
        pos = view.split_local(i, plan["left_text"], plan["split_s"],
                               {"id": ids["insert_id"],
                                "payload": {"operation": "chunk_insert",
                                            "after_segment_id": plan["after_id"],
                                            "start_time": plan["split_s"],
                                            "end_time": plan["end_s"],
                                            "label": None,
                                            "text": plan["right_text"]}})
        # Register the group so x on the fresh right half UNSPLITS without a
        # reload (the ac84360a marker, cashed in by action_remove_insert).
        view.split_groups[ids["insert_id"]] = {
            "group_ids": [ids["text_id"], ids["nudge_id"]],
            "target_id": plan["segment_id"], "old_text": old_text,
            "old_end": plan["end_s"]}
        if pos is not None:
            # _marks is POSITIONAL: positions at/after the splice shift right
            # (the insert echo's indexing perturbation, DEC 3d3fa2a8 known cost).
            self._marks = {(k + 1 if k >= pos else k): v for k, v in self._marks.items()}
        self._render()
        status.update(f"✂ split #{seg.index} at {plan['split_s']:.2f}s (caret-seeded)"
                      " — ,/. tunes the new seam · g auditions it")

    def action_relabel_insert(self) -> None:
        """L: relabel the inserted chunk under the cursor (drive ask
        2026-07-29: accepted 'inhale' proposals that turn out to be silence
        need a one-gesture reclassification — 'empty' tightens speech chunks
        without polluting the inhale class; 'empty' is THE hard-negative
        term, user-ratified 2026-07-30 over the 'dead-air' synonym, finding
        8c0aa0bf). Opens the I-editor menu pre-filled with the CURRENT label;
        commits as supersede + re-insert (same span, same anchors, new
        label) — no new op vocabulary, and the bench derives it as a
        RELABELED verdict. Split right halves refuse (their label is
        structural, x unsplits them)."""
        view = self.view
        seg = view.segments[self.cursor]
        status = self.query_one("#status", Static)
        if seg.id not in view.inserted_ids:
            status.update("relabel: only inserted (⊕) chunks carry labels")
            return
        if seg.id in view.split_groups:
            status.update("relabel: a split right half has no class label (x unsplits)")
            return
        editor = self.query_one("#editor", Input)
        self._input_mode = "relabel"
        editor.value = f"{view.insert_labels.get(seg.id) or self._insert_label} "
        editor.display = True
        editor.focus()
        menu = self._insert_label_menu()
        status.update("relabel: class-or-# · "
                      + " ".join(f"{i + 1}:{c}" for i, c in enumerate(menu)))

    async def _submit_relabel(self, raw: str) -> None:
        """The L-editor submission: supersede the insert and re-commit it with
        the new label — identical span/anchors/rank, so the projection and the
        bench see the same event under its corrected class."""
        self._close_editor()
        view = self.view
        status = self.query_one("#status", Static)
        raw, err = resolve_mark_class_token(raw, self._insert_label_menu())
        if err:
            self._render()
            status.update(f"relabel: {err}")
            return
        tokens = (raw or "").split()
        if not tokens:
            self._render()
            return
        label = tokens[0].strip('`"\'')
        if not label or not label[:1].isalnum():
            self._render()
            status.update("relabel: label must start with a letter or digit")
            return
        i = self.cursor
        seg = view.segments[i]
        if seg.id not in view.inserted_ids or seg.start_time is None:
            self._render()
            status.update("relabel: cursor moved off the insert — try again")
            return
        old_label = view.insert_labels.get(seg.id)
        if label == old_label:
            self._render()
            status.update(f"relabel: already [{label}]")
            return
        start_s, end_s = float(seg.start_time), float(seg.end_time)
        rank = view.insert_ranks.get(seg.id, 0.0)
        text = seg.text or ""
        # anchors re-derive like plan_chunk_insert: nearest layer-0 flanks
        after_id = next((view.segments[j].id for j in range(i - 1, -1, -1)
                         if view.segments[j].id not in view.inserted_ids), None)
        before_id = next((view.segments[j].id for j in range(i + 1, view.size)
                          if view.segments[j].id not in view.inserted_ids), None)
        if after_id is None:
            self._render()
            status.update("relabel: no layer-0 anchor left of the insert")
            return
        await commit_chunk_insert_removal(
            view.queue, view.graph_id, view.source_id, seg.id,
            self.session_id, actor=self.actor, journal_path=self._journal_path)
        view.remove_insert_local(seg.id)
        insert_id = await commit_chunk_insert_correction(
            view.queue, view.graph_id, view.source_id,
            after_id, start_s, end_s, self.session_id,
            before_segment_id=before_id, label=label,
            rank=rank, actor=self.actor, journal_path=self._journal_path)
        pos = view.add_insert_local(
            {"id": insert_id,
             "payload": {"operation": "chunk_insert", "after_segment_id": after_id,
                         "start_time": start_s, "end_time": end_s,
                         "label": label, "text": text, "rank": rank}})
        if pos is not None:
            self.cursor = pos
        view.refresh_event_proposals()
        self._render()
        status.update(f"↺ relabeled [{old_label or '∅'}] → [{label}]"
                      f" ({start_s:.2f}–{end_s:.2f}s)")

    async def action_remove_insert(self) -> None:
        """x: remove the inserted chunk under the cursor (reject-as-supersede,
        the mark-dismissal pattern) — a mistaken one-keystroke insert needs a
        one-keystroke out. A SPLIT's right half UNSPLITS instead: superseding
        only the insert would orphan its text and leave the target truncated
        at the cut (FINDING 131ba57a follow-on — the retyping cost), so x
        supersedes the whole split group and the target gets its pre-split
        text and end back. Layer-0 segments refuse: nothing else is removable."""
        view = self.view
        seg = view.segments[self.cursor]
        status = self.query_one("#status", Static)
        if seg.id not in view.inserted_ids:
            status.update("remove: only inserted (⊕) chunks can be removed")
            return
        info = view.split_groups.get(seg.id)
        if info:
            await commit_chunk_split_removal(
                view.queue, view.graph_id, view.source_id, seg.id,
                info["group_ids"], self.session_id, actor=self.actor,
                journal_path=self._journal_path)
            pos = view.unsplit_local(seg.id)
        else:
            await commit_chunk_insert_removal(
                view.queue, view.graph_id, view.source_id, seg.id,
                self.session_id, actor=self.actor, journal_path=self._journal_path)
            pos = view.remove_insert_local(seg.id)
        if pos is not None:
            self._marks = {(k - 1 if k > pos else k): v
                           for k, v in self._marks.items() if k != pos}
            self.cursor = max(0, min(view.size - 1, self.cursor))
        # An x-removed accept gives its proposal BACK (pending derives from
        # the raw set vs spine state — drive find 2026-07-29).
        view.refresh_event_proposals()
        self._render()
        if info:
            status.update(f"⊖ unsplit: right half removed, target restored"
                          f" (text + end back to {seg.end_time:.2f}s)")
        else:
            status.update(f"⊖ removed inserted chunk"
                          f" ({seg.start_time:.2f}–{seg.end_time:.2f}s)")

    async def _play_source_span(self, start_s: float, end_s: float,
                                note: str = "") -> None:
        """Decode + play a source-coordinate span of the ORIGINAL media — the
        inserted-chunk playback path (no model-input WAV need cover it). The
        live ticker rides every span play; `note` names what is sounding."""
        path = self.view.source_path
        status = self.query_one("#status", Static)
        if not path or not Path(path).exists():
            status.update(f"insert audio: source media not found ({path or 'no path on Source'})")
            return
        try:
            samples = await load_source_slice(path, start_s, end_s,
                                              samplerate=self.player.samplerate)
        except (RuntimeError, OSError) as e:
            status.update(f"insert audio: decode failed — {e}")
            return
        if self.speed != 1.0 and len(samples):
            samples = stretch(samples, self.speed)
        self.player.play(samples)
        self._start_ticker(start_s, end_s, note)

    async def _shift_boundary(self, direction: str) -> None:
        """One [ / ] press: move ONE word across the boundary AFTER the cursor.

        Commits a boundary_shift Correction (word-level payload, layer 0.0.8
        semantics); when the RECEIVING segment is prune-covered, also commits
        the unprune amendment (the falsified-D14 rescue — without it the
        projection drops the moved text with the pruned position). Key-repeat
        is DROPPED while a commit is in flight, so a held key can only shift
        as fast as the screen shows it (first-drive feedback, 2026-07-12)."""
        now = time.monotonic()
        if self._shift_busy or now - self._last_shift < self._shift_floor:
            return  # busy commit OR inside the paint-rate floor — drop the repeat
        self._shift_busy = True
        try:
            await self._shift_boundary_now(direction)
        finally:
            self._last_shift = time.monotonic()
            self._shift_busy = False

    async def _shift_boundary_now(self, direction: str) -> None:
        view, i = self.view, self.cursor
        status = self.query_one("#status", Static)
        if i + 1 >= view.size:
            status.update("boundary shift: no segment after the cursor")
            return
        if view.segments[i].id in view.inserted_ids:
            status.update("boundary shift: ✋ inserted chunk — its text lives on the overlay (e edits it)")
            return
        # Partner resolution (DEC a5754fa4): hop certified-wordless inserts to
        # the next LAYER-0 segment — the committed op is the same layer-0
        # boundary_shift the old x-remove/shift/re-accept dance produced,
        # without the dance (the moved word's audio was already across the
        # inserts: FA misplacement, often CAUSED by the intervening event).
        # A word-bearing insert between still refuses — shifting past it
        # would reorder spoken words on the effective spine.
        j = i + 1
        while j < view.size and self._wordless_insert(j):
            j += 1
        if j >= view.size:
            status.update("boundary shift: no layer-0 segment after the cursor")
            return
        if view.segments[j].id in view.inserted_ids:
            status.update("boundary shift: ✋ word-bearing insert between — its text lives on the overlay (e edits it)")
            return
        if view.aseg_index(i) != view.aseg_index(j):
            status.update("boundary shift: ✋ audio-segment seam — text stays within its audio segment")
            return
        left, right = view.segments[i], view.segments[j]
        plan = plan_boundary_shift(left.text, right.text, direction)
        if plan is None:
            status.update(f"boundary shift: nothing to {direction}")
            return
        moved, new_left, new_right = plan
        await commit_boundary_shift_correction(
            view.queue, view.graph_id, view.source_id, left.id, right.id,
            moved, direction, self.session_id, actor=self.actor,
            journal_path=self._journal_path)
        receiver = right if direction == "push" else left
        if receiver.id in view.pruned_ids:
            prior = view.prune_correction_for(receiver.id)
            if prior is not None:
                amended = await commit_prune_amendment(
                    view.queue, view.graph_id, prior, [receiver.id],
                    self.session_id, actor=self.actor,
                    journal_path=self._journal_path)
                view.unprune_local(prior["id"], amended)
        left.text, right.text = new_left, new_right   # local echo (same math as the layer)
        self._marks[i] = "corrected"
        self._marks[j] = "corrected"
        self._render()

    async def action_shift_push(self) -> None:
        await self._shift_boundary("push")

    async def action_shift_pull(self) -> None:
        await self._shift_boundary("pull")

    def _jump_glyph(self, direction: int, ids: set, what: str) -> None:
        """n/N (⚑), p/P (✂): cursor to the next/previous segment in a glyph id
        set (wraps) — resolution passes walk glyphs directly instead of
        scanning thousands of segments (drive find: had to dig for a ✂)."""
        view = self.view
        if not ids:
            self.query_one("#status", Static).update(f"no {what} segments on this source")
            return
        for step in range(1, view.size + 1):
            j = (self.cursor + direction * step) % view.size
            if view.segments[j].id in ids:
                self._move(j - self.cursor)
                return

    def action_next_mark(self) -> None:
        self._jump_glyph(1, self.view.marked_ids, "⚑ marked")

    def action_prev_mark(self) -> None:
        self._jump_glyph(-1, self.view.marked_ids, "⚑ marked")

    def action_next_prune(self) -> None:
        self._jump_glyph(1, self.view.pruned_ids, "✂ pruned")

    def action_prev_prune(self) -> None:
        self._jump_glyph(-1, self.view.pruned_ids, "✂ pruned")

    def _mark_class_menu(self) -> List[str]:
        """Selectable classes for the M picker: the recommended slate first, then
        classes carried by this source's OPEN marks — dismissing a class's last
        open mark removes it (junk cleanup); proven classes persist by promotion
        into RECOMMENDED_MARK_CLASSES (open vocab, DEC 2a231843)."""
        return list(RECOMMENDED_MARK_CLASSES) + [
            c for c in self.view.seen_mark_classes if c not in RECOMMENDED_MARK_CLASSES]

    async def action_mark_quick(self) -> None:
        """m: mark the focused segment with the last-used class and keep walking —
        the held-back-corrections gesture (DEC 42854519) must cost one keystroke."""
        seg = self.view.segments[self.cursor]
        await self._commit_mark({"kind": "segment", "segment_id": seg.id},
                                self._mark_class, None)

    async def action_mark_boundary(self) -> None:
        """b: mark the boundary AFTER the cursor (the shift gesture's coordinates).

        Unlike shifts, audio-segment seams are NOT refused — a suspect seam is
        exactly what a boundary mark is for."""
        view, i = self.view, self.cursor
        if i + 1 >= view.size:
            self.query_one("#status", Static).update("boundary mark: no segment after the cursor")
            return
        await self._commit_mark({"kind": "boundary",
                                 "boundary_after": view.segments[i].id,
                                 "right_segment_id": view.segments[i + 1].id},
                                self._mark_class, None)

    def action_mark_editor(self) -> None:
        """M: class-picker mark — `class-or-# ["snippet"] [note...]`: a leading
        digit picks from the numbered class menu (recommended slate + this
        source's journaled classes); a snippet found in the segment text becomes
        a SPAN anchor; a punctuation-led token dismisses ALL open marks at the
        cursor."""
        editor = self.query_one("#editor", Input)
        self._input_mode = "mark"
        editor.value = f"{self._mark_class} "
        editor.display = True
        editor.focus()
        menu = self._mark_class_menu()
        self.query_one("#status", Static).update(
            'mark: class-or-# ["snippet"] [note] · - dismiss · '
            + " ".join(f"{i + 1}:{c}" for i, c in enumerate(menu)))

    async def _submit_mark(self, raw: str) -> None:
        seg = self.view.segments[self.cursor]
        self._close_editor()
        tokens = raw.split()
        if not tokens:
            self._render()
            return
        first = tokens[0].strip('`"\'')
        if first.startswith("-") or not first:
            # Dismissal gesture, tolerant of formatting fumbles ('`-`', '- oops'):
            # a punctuation-led token must never mint a junk class and hijack the
            # last-used class (drive find, 2026-07-19). ALL open marks at the
            # cursor go — the ⚑ must actually clear (boundary marks from a
            # neighbor's b press anchor this segment too).
            marks = self.view.marks_for(seg.id)
            if not marks:
                self._render()
                self.query_one("#status", Static).update(f"no open mark on #{seg.index}")
                return
            for m in marks:
                await commit_mark_dismissal(
                    self.view.queue, self.view.graph_id, self.view.source_id,
                    m["id"], self.session_id, actor=self.actor,
                    journal_path=self._journal_path)
                self.view.dismiss_mark_local(m["id"])
            classes = ", ".join(str((m.get("payload") or {}).get("mark_class")) for m in marks)
            self._render()
            self.query_one("#status", Static).update(
                f"dismissed {len(marks)} mark(s) on #{seg.index} [{classes}]")
            return
        raw, err = resolve_mark_class_token(raw, self._mark_class_menu())
        if err:
            self._render()
            self.query_one("#status", Static).update(f"mark: {err}")
            return
        parsed = parse_mark_input(raw, seg.text)
        if parsed is None:
            self._render()
            return
        mark_class, span, note = parsed
        if span is not None:
            start, end, snapshot = span
            anchor = {"kind": "span", "segment_id": seg.id, "char_start": start,
                      "char_end": end, "text_snapshot": snapshot}
        else:
            anchor = {"kind": "segment", "segment_id": seg.id}
        await self._commit_mark(anchor, mark_class, note)

    async def _commit_mark(self, anchor: Dict[str, Any], mark_class: str,
                           note: Optional[str]) -> None:
        """Commit one mark Correction + local echo (the ⚑ paints immediately).

        A mark records attention, not a decision: no review marker, no text
        change, the cursor stays put — mark and keep walking."""
        try:
            mark_id = await commit_mark_correction(
                self.view.queue, self.view.graph_id, self.view.source_id,
                anchor, mark_class, self.session_id, actor=self.actor, note=note,
                journal_path=self._journal_path)
        except ValueError as e:
            self._render()
            self.query_one("#status", Static).update(f"mark refused: {e}")
            return
        self._mark_class = mark_class
        save_tui_state(self._graph_db_path, self.view.source_id, self.cursor,
                       mark_class=mark_class)
        self.view.add_mark_local({"id": mark_id, "correction_type": "mark",
                                  "payload": {"operation": "mark", "anchor": dict(anchor),
                                              "mark_class": mark_class}})
        self._render()
        seg = self.view.segments[self.cursor]
        suffix = f" — {note}" if note else ""
        self.query_one("#status", Static).update(
            f"⚑ #{seg.index} [{mark_class}] ({anchor['kind']}){suffix}")

    def action_gate_editor(self) -> None:
        """F: the extraction-gate editor (DEC 8e05b87b, flywheel build leg 1) —
        `w [sec]` watermarks the pause point (pre-filled with the cursor
        segment's end: pausing a pass = F, enter), `signoff` = signed_off with
        the watermark at end-of-source, `exclude`/`resume` flip the status.
        Every submit is one journaled append-only assertion on THIS spine."""
        editor = self.query_one("#editor", Input)
        self._input_mode = "gate"
        seg = self.view.segments[self.cursor]
        editor.value = (f"w {float(seg.end_time):.1f}"
                        if seg.end_time is not None else "w ")
        editor.display = True
        editor.focus()
        self.query_one("#status", Static).update(
            "gate: w [sec] watermark-at-pause · signoff · exclude · resume"
            + f" · now: {self._gate_chip() or 'in_progress (default), no watermark'}")

    def _gate_chip(self) -> str:
        """The status-strip gate chip: empty when never asserted (the quiet
        in_progress default), else status glyph + watermark seconds."""
        gate = self.view.gate if self.view is not None else None
        if not gate:
            return ""
        glyph = {"in_progress": "▶", "signed_off": "✔", "excluded": "✘"}.get(
            str(gate.get("extraction_status")), "?")
        wm = gate.get("annotated_through")
        return f"gate {glyph}{f'{float(wm):.1f}s' if wm is not None else ''}"

    async def _submit_gate(self, raw: str) -> None:
        """The F-editor submission: plan (pure) + commit ONE gate assertion +
        local echo. The watermark is asserted EXPLICITLY — never derived from
        op positions (DEC 8e05b87b: derivation underestimates under op-free
        paging)."""
        self._close_editor()
        view = self.view
        status = self.query_one("#status", Static)
        ends = [float(s.end_time) for s in view.segments if s.end_time is not None]
        seg = view.segments[self.cursor]
        plan = plan_gate(raw,
                         float(seg.end_time) if seg.end_time is not None else None,
                         max(ends) if ends else None,
                         (view.gate or {}).get("annotated_through"))
        if plan is None:
            self._render()
            status.update("gate: w [sec] · signoff · exclude · resume "
                          "(refused — unknown verb or no time to anchor)")
            return
        new_status, watermark = plan
        gate_id = await commit_extraction_gate(
            view.queue, view.graph_id, view.source_id, view.skeleton_hash,
            new_status, watermark, session_id=self.session_id,
            actor=self.actor, journal_path=self._journal_path)
        view.set_gate_local({"id": gate_id, "source_id": view.source_id,
                             "skeleton_hash": view.skeleton_hash,
                             "extraction_status": new_status,
                             "annotated_through": watermark,
                             "actor": self.actor, "created_at": time.time()})
        self._render()
        wm_txt = f"{float(watermark):.1f}s" if watermark is not None else "none"
        status.update(f"⛭ gate asserted: {new_status} · annotated_through {wm_txt}")

    def _resolve_fa_cache(self) -> Optional[Path]:
        """Resolve the forced-alignment cache db (the annotate lane's word-time
        source): explicit --fa-cache-db, else the workspace's FA capability
        cache (the scan-mishomed default). None = estimation-only snapping."""
        if self._fa_cache_arg:
            p = Path(self._fa_cache_arg)
            return p if p.is_file() else None
        ws = resolve_workspace(explicit=None)
        if ws is None:
            return None
        p = (ws.substrate_data_dir / "data" / "cjm-capability-qwen3-forced-aligner"
             / "forced_alignments.db")
        return p if p.is_file() else None

    def _overlay_label_menu(self) -> List[str]:
        """Selectable overlay labels: the recommended slate first, then labels
        carried by this source's ACTIVE overlays (the mark-class menu pattern —
        open vocabulary, slate is DATA)."""
        return list(RECOMMENDED_OVERLAY_LABELS) + [
            c for c in self.view.seen_overlay_labels
            if c not in RECOMMENDED_OVERLAY_LABELS]

    def _selection_range(self, n_tokens: int) -> Optional[Tuple[int, int]]:
        """The selected token range (inclusive), clamped: the v-anchor..cursor
        span when a range is anchored, else the word under the cursor. None =
        no words."""
        if n_tokens <= 0:
            return None
        c = max(0, min(n_tokens - 1, self._word_cursor))
        if self._word_anchor is None:
            return c, c
        a = max(0, min(n_tokens - 1, self._word_anchor))
        return (min(a, c), max(a, c))

    def _annotate_body(self, seg) -> Text:
        """The focused card's word-level paint in the annotate lane: the word
        cursor underlined, the v-selection yellow, committed overlay spans
        cyan (span offsets from each overlay's text-indexed anchor). Spans
        only — no base style (7aca1117)."""
        toks = segment_word_tokens(seg.text)
        committed = []
        for o in self.view.overlays_for(seg.id):
            a = (o.get("payload") or {}).get("anchor") or {}
            if a.get("char_start") is not None and a.get("char_end") is not None:
                committed.append((int(a["char_start"]), int(a["char_end"])))
        sel = self._selection_range(len(toks))
        body = Text()
        for i, (cs, ce, w) in enumerate(toks):
            if i:
                body.append(" ")
            style = ""
            if any(cs >= a and ce <= b for a, b in committed):
                style = "cyan"
            if sel is not None and sel[0] <= i <= sel[1]:
                style = "bold yellow"
            if i == self._word_cursor:
                style = (style + " underline").strip()
            body.append(w, style or None)
        return body

    async def _fa_words_for(self, seg) -> Optional[List[Dict[str, Any]]]:
        """The segment's transcript FA words (source seconds), memoized per
        Transcript node. None = no cache / join miss (estimation fallback)."""
        tid = getattr(seg, "text_from", None)
        if not tid or self._fa_cache_db is None:
            return None
        if tid not in self._fa_words_cache:
            self._fa_words_cache[tid] = await fa_words_for_transcript(
                self.view.queue, self.view.graph_id, tid, self._fa_cache_db)
        return self._fa_words_cache[tid]

    async def _snap_selection(self, seg) -> Optional[Dict[str, Any]]:
        """Derive the current selection's FA-snapped record fields; None (with
        a painted status) = refused."""
        status = self.query_one("#status", Static)
        toks = segment_word_tokens(seg.text)
        sel = self._selection_range(len(toks))
        if sel is None:
            status.update("annotate: segment has no words (e-edit text lands in the walk lane)")
            return None
        if seg.start_time is None or seg.end_time is None:
            status.update("annotate: segment has no audio times to snap against")
            return None
        a, b = sel
        snapped = snap_word_span(toks, a, b, float(seg.start_time), float(seg.end_time),
                                 len(seg.text), await self._fa_words_for(seg))
        if snapped is None:
            status.update("annotate: selection refused (word range invalid)")
            return None
        start_s, end_s, snap, words = snapped
        char_start, char_end = toks[a][0], toks[b][1]
        return {"char_start": char_start, "char_end": char_end,
                "text": seg.text[char_start:char_end],
                "start_time": start_s, "end_time": end_s,
                "snap": snap, "words": words}

    def _word_move(self, delta: int) -> None:
        seg = self.view.segments[self.cursor]
        n = len(segment_word_tokens(seg.text))
        if n == 0:
            self.query_one("#status", Static).update(
                "annotate: no words on this segment — j/k to a text-bearing one")
            return
        self._word_cursor = max(0, min(n - 1, self._word_cursor + delta))
        self._render()

    def action_word_left(self) -> None:
        self._word_move(-1)

    def action_word_right(self) -> None:
        self._word_move(1)

    def action_word_select(self) -> None:
        """v: anchor/clear the word-range selection (the vim visual gesture) —
        anchored, h/l extends the range; v again collapses it."""
        seg = self.view.segments[self.cursor]
        n = len(segment_word_tokens(seg.text))
        if n == 0:
            self.query_one("#status", Static).update("annotate: no words to select")
            return
        self._word_anchor = None if self._word_anchor is not None \
            else max(0, min(n - 1, self._word_cursor))
        self._render()

    async def action_annotate_audition(self) -> None:
        """R (annotate lane): audition the selection's FA-snapped span — hear
        exactly what a commit would record before recording it."""
        seg = self.view.segments[self.cursor]
        rec = await self._snap_selection(seg)
        if rec is None:
            return
        self.player.stop()
        self.run_worker(self._play_source_span(
            rec["start_time"], rec["end_time"],
            note=f" · ◈? “{rec['text'][:24]}” ({rec['snap']}) · space/1-9 commits"))

    async def action_annotate_quick(self) -> None:
        """space (annotate lane): commit the selection under the LAST-USED
        label — the one-keystroke sample drive (the assign-space precedent)."""
        await self._commit_overlay(self._overlay_label, None)

    async def action_annotate_pick(self, n: int) -> None:
        """1-9 (annotate lane): commit the selection under menu label #n."""
        menu = self._overlay_label_menu()
        if not (1 <= n <= len(menu)):
            self.query_one("#status", Static).update(
                f"annotate: no label #{n} — menu has {len(menu)} (A types a new one)")
            return
        await self._commit_overlay(menu[n - 1], None)

    def action_annotate_editor(self) -> None:
        """A (annotate lane): the open-vocabulary label editor —
        `label-or-# [note...]` (the M/I picker grammar, shared resolver)."""
        editor = self.query_one("#editor", Input)
        self._input_mode = "annotate"
        editor.value = f"{self._overlay_label} "
        editor.display = True
        editor.focus()
        menu = self._overlay_label_menu()
        self.query_one("#status", Static).update(
            "annotate label: class-or-# [note] · "
            + " ".join(f"{i + 1}:{c}" for i, c in enumerate(menu)))

    async def _submit_annotate(self, raw: str) -> None:
        """The A-editor submission: first token = the overlay label (digit
        resolves against the menu), the rest is the note."""
        self._close_editor()
        raw, err = resolve_mark_class_token(raw, self._overlay_label_menu())
        if err:
            self._render()
            self.query_one("#status", Static).update(f"annotate: {err}")
            return
        tokens = (raw or "").split()
        if not tokens:
            self._render()
            return
        label = tokens[0].strip('`"\'')
        if not label or not label[:1].isalnum():
            self._render()
            self.query_one("#status", Static).update(
                "annotate: label must start with a letter or digit")
            return
        await self._commit_overlay(label, " ".join(tokens[1:]) or None)

    async def _commit_overlay(self, label: str, note: Optional[str]) -> None:
        """Commit one speech overlay from the current selection: FA-snap the
        word range, journal the sample, echo the ◈, and AUDITION the recorded
        span (the toolkit's immediate-audible-verification regime)."""
        view = self.view
        seg = view.segments[self.cursor]
        rec = await self._snap_selection(seg)
        if rec is None:
            self._render()
            return
        anchor = {"kind": "span", "segment_id": seg.id,
                  "char_start": rec["char_start"], "char_end": rec["char_end"],
                  "text_snapshot": rec["text"]}
        try:
            overlay_id = await commit_speech_overlay_correction(
                view.queue, view.graph_id, view.source_id, anchor, label,
                rec["start_time"], rec["end_time"], rec["text"], self.session_id,
                words=rec["words"], snap=rec["snap"], actor=self.actor, note=note,
                journal_path=self._journal_path)
        except ValueError as e:
            self._render()
            self.query_one("#status", Static).update(f"annotate refused: {e}")
            return
        view.add_overlay_local({"id": overlay_id, "correction_type": "annotation",
                                "payload": {"operation": "speech_overlay",
                                            "anchor": dict(anchor), "label": label,
                                            "start_time": rec["start_time"],
                                            "end_time": rec["end_time"],
                                            "text": rec["text"], "snap": rec["snap"]}})
        self._overlay_label = label
        save_tui_state(self._graph_db_path, view.source_id, self.cursor,
                       overlay_label=label)
        self._word_anchor = None   # committed — the range hands back to the cursor
        self._render()
        self.player.stop()
        self.run_worker(self._play_source_span(
            rec["start_time"], rec["end_time"],
            note=f" · ◈ {label} “{rec['text'][:24]}” ({rec['snap']})"))

    def _overlay_at_cursor(self, seg) -> Optional[Dict[str, Any]]:
        """The gesture-target overlay on the cursor segment: the one covering
        the word cursor when there is one, else the newest; None = no ◈."""
        overlays = self.view.overlays_for(seg.id)
        if not overlays:
            return None
        toks = segment_word_tokens(seg.text)
        if toks:
            c = max(0, min(len(toks) - 1, self._word_cursor))
            cs, ce = toks[c][0], toks[c][1]
            for o in reversed(overlays):
                a = (o.get("payload") or {}).get("anchor") or {}
                if a.get("char_start") is not None and a.get("char_end") is not None \
                        and int(a["char_start"]) <= cs and ce <= int(a["char_end"]):
                    return o
        return overlays[-1]

    async def action_overlay_nudge(self, edge: str, sign: int) -> None:
        """,/. (span END) and </> (span START) in the annotate lane: nudge the
        cursor overlay's TIME span by ±the { } ladder step — the fc42614d
        refinement half (snap-to-word derives, nudges refine; drive ask
        2026-08-03: an estimated 'uh' span needed its tail grown).

        Refinement is a SUPERSEDE + re-commit (same anchor/label/text/words,
        new times, snap="nudged" — human-refined provenance the bench can
        split from machine snaps): overlays are samples, never spine timing,
        so the time-nudge verb stays out of it and the supersede CHAIN is the
        refinement record. The adjusted span replays at once (the immediate-
        audible-verification regime)."""
        view = self.view
        seg = view.segments[self.cursor]
        status = self.query_one("#status", Static)
        target = self._overlay_at_cursor(seg)
        if target is None:
            status.update("annotate: no ◈ overlay here to nudge (v+label commits one first)")
            return
        p = target.get("payload") or {}
        start, end = float(p["start_time"]), float(p["end_time"])
        delta = sign * self._nudge_step
        if edge == "end":
            new_start, new_end = start, end + delta
        else:
            new_start, new_end = max(0.0, start + delta), end
        if new_end - new_start < 0.005:
            status.update(f"annotate: nudge refused ({edge} {delta:+.3f}s would "
                          "collapse the span)")
            return
        anchor = dict(p.get("anchor") or {})
        overlay_id = await commit_speech_overlay_correction(
            view.queue, view.graph_id, view.source_id, anchor,
            str(p.get("label")), new_start, new_end, str(p.get("text") or ""),
            self.session_id, words=list(p.get("words") or []), snap="nudged",
            supersedes_id=target["id"], actor=self.actor,
            journal_path=self._journal_path)
        view.remove_overlay_local(target["id"])
        view.add_overlay_local({"id": overlay_id, "correction_type": "annotation",
                                "payload": {**p, "anchor": anchor,
                                            "start_time": new_start,
                                            "end_time": new_end, "snap": "nudged"}})
        self._render()
        self.player.stop()
        self.run_worker(self._play_source_span(
            new_start, new_end,
            note=f" · ◈ {p.get('label')} {edge} {delta:+.3f}s"))

    async def action_overlay_remove(self) -> None:
        """x (annotate lane): remove an overlay on the cursor segment — the one
        covering the word cursor when there is one, else the newest
        (reject-as-supersede, the mark-dismissal shape)."""
        view = self.view
        seg = view.segments[self.cursor]
        status = self.query_one("#status", Static)
        target = self._overlay_at_cursor(seg)
        if target is None:
            status.update("annotate: no ◈ overlay on this segment")
            return
        await commit_speech_overlay_removal(
            view.queue, view.graph_id, view.source_id, target["id"],
            self.session_id, actor=self.actor, journal_path=self._journal_path)
        view.remove_overlay_local(target["id"])
        self._render()
        p = target.get("payload") or {}
        status.update(f"⊘ removed ◈ [{p.get('label')}] “{str(p.get('text') or '')[:24]}”")

    def action_next_overlay(self) -> None:
        self._jump_glyph(1, self.view.overlay_ids, "◈ annotated")

    def action_prev_overlay(self) -> None:
        self._jump_glyph(-1, self.view.overlay_ids, "◈ annotated")

    def action_cancel(self) -> None:
        editor = self.query_one("#editor", Input)
        if editor.display:
            self._accept_cluster = None  # an aborted a-gesture must not hijack the next assign
            self._pending_proposal = None  # an aborted propose-split hop likewise
            self._close_editor()
            self._render()
        elif self._word_anchor is not None:
            self._word_anchor = None     # esc clears the word selection first
            self._render()
        else:
            self.player.stop()
            self._stop_ticker()
            self._render()  # esc hands the status line back to the lane

    async def action_quit_app(self) -> None:
        if self.view is not None:
            save_tui_state(self._graph_db_path, self.view.source_id, self.cursor,
                           speed=self.speed)
        if self.player is not None:
            self.player.close()
        if self.view is not None:
            await self.view.close()
        elif self._queue is not None:
            await self._queue.stop()   # picker-stage quit: the stack is open, no view owns it yet
        self.exit()


def load_tui_state(
    graph_db_path: str,  # The graph db whose sidecar state file to read
) -> Dict[str, Any]:  # {source_id: {"cursor": int, "ts": float}}; empty when absent/corrupt
    """Read the per-graph TUI sidecar state (last-focused positions)."""
    return SidecarState(f"{graph_db_path}.tui-state.json").load()


def save_tui_state(
    graph_db_path: str,  # The graph db whose sidecar state file to write
    source_id: str,      # Source whose position is being remembered
    cursor: Optional[int],  # Last-focused segment position (None = leave as-is)
    speed: Optional[float] = None,  # Playback-rate preference (db-wide `_speed`; None = leave as-is)
    mark_class: Optional[str] = None,  # Last-used ⚑ class (db-wide `_mark_class`; None = leave as-is)
    insert_label: Optional[str] = None,  # Last-used ⊕ insert label (db-wide `_insert_label`; None = leave as-is)
    overlay_label: Optional[str] = None,  # Last-used ◈ overlay label (db-wide `_overlay_label`; None = leave as-is)
    nudge_step_ms: Optional[float] = None,  # Nudge-step preference (db-wide `_nudge_step_ms`; None = leave as-is)
    lane: Optional[str] = None,      # Pass-lane preference (db-wide `_lane`; None = leave as-is)
    fold_wordless: Optional[bool] = None,  # z fold preference (db-wide `_fold_wordless`; None = leave as-is)
    skeleton: Optional[str] = None,  # Chosen skeleton-spine selector (per-source; None = leave as-is)
    spines: Optional[int] = None,    # Spine-set size the choice was made against (re-prompt key)
) -> None:
    """Merge one source's view state into the sidecar state file.

    VIEW state, not knowledge — it lives in a local sidecar next to the db,
    never as a graph write (the cursor is where the eye was, not a decision;
    the spine CHOICE is a view preference too — the graph-asserted active
    spine stays deferred per DEC f1024568). Per-source entries MERGE so a
    cursor write never drops the spine choice and vice versa. Write failures
    are silently tolerated: losing a bookmark must never break the loop."""
    store = SidecarState(f"{graph_db_path}.tui-state.json")
    state = store.load()
    entry = dict(state.get(source_id) or {})
    if cursor is not None:
        entry["cursor"] = int(cursor)
    entry["ts"] = time.time()
    if skeleton is not None:
        entry["skeleton"] = str(skeleton)
    if spines is not None:
        entry["spines"] = int(spines)
    state[source_id] = entry
    if speed is not None:
        state["_speed"] = float(speed)
    if mark_class is not None:
        state["_mark_class"] = str(mark_class)
    if insert_label is not None:
        state["_insert_label"] = str(insert_label)
    if overlay_label is not None:
        state["_overlay_label"] = str(overlay_label)
    if nudge_step_ms is not None:
        state["_nudge_step_ms"] = float(nudge_step_ms)
    if lane is not None:
        state["_lane"] = str(lane)
    if fold_wordless is not None:
        state["_fold_wordless"] = bool(fold_wordless)
    store.write(state)


def spine_label(
    spine: Dict[str, Any],  # One list_source_spines row
) -> str:  # Picker-row config summary
    """One picker row's config summary for a skeleton spine (pure).

    Legacy (no skeleton_hash) reads as the incumbent VAD-only spine; split
    spines show their policy tag + a hash prefix (the persisted selector value
    stays the FULL hash — see selector_for_spine)."""
    h = spine.get("skeleton_hash")
    if not h:
        return "vad-only (pre-split)"
    tag = spine.get("split_policy") or "vad-only"
    return f"{tag} · {str(h).split(':')[-1][:8]}"


def selector_for_spine(
    spine: Dict[str, Any],  # One list_source_spines row
) -> str:  # The --skeleton selector naming this spine
    """The selector value a picker choice persists (pure): the full skeleton
    hash, or the LEGACY_SKELETON token for the pre-split spine."""
    return str(spine.get("skeleton_hash") or LEGACY_SKELETON)
