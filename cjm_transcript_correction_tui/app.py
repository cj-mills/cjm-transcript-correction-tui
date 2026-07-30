import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_context_graph_layer.journal import sidecar_journal_path
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
                                                  commit_speaker_entity, commit_text_correction,
                                                  commit_time_nudge_correction, LEGACY_SKELETON,
                                                  list_source_spines, list_speaker_entities,
                                                  session_purposes_by_source, start_session)
from cjm_transcript_correction_core.models import (RECOMMENDED_INSERT_LABELS,
                                                   RECOMMENDED_MARK_CLASSES)
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, Static

from .spine import (list_sources, load_source_slice, match_sources, open_stack, parse_entity_input,
                    parse_mark_input, plan_boundary_shift, plan_chunk_insert, plan_chunk_split,
                    plan_gate, plan_time_nudge, resolve_mark_class_token, source_status, SpineView)


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

    NUDGE_STEPS_MS = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)  # the { } nudge-step ladder (first drive: 100ms fits some cuts, others need 20/10/5; 1ms added 2026-07-27 — a Chris Wright boundary at ~358.5s outgrew 5ms — granularity is per-BOUNDARY)

    NUDGE_TAIL_S = 2.0  # Max seconds of segment TAIL an end-nudge replays (the edge under judgment, not the whole segment)

    # Lane vocabulary (DEC cc55a7b5 multi-lane workbench, v1 = walk + assign per
    # 8a4df244): the assign lane exposes ONLY these actions; assign-only actions
    # are inert in the walk lane. One data table, one check_action gate.
    ASSIGN_LANE_ACTIONS = frozenset({
        "next", "prev", "replay", "seam_next", "seam_prev", "speed_down", "speed_up",
        "yank", "assign_pick", "assign_same", "assign_new", "assign_accept",
        "cycle_lane", "cancel", "quit_app"})
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
        "insert_chunk", "insert_labeled", "remove_insert",
        "propose_accept", "propose_next", "propose_prev",
        "cycle_lane", "cancel", "quit_app"})
    PROPOSE_ONLY_ACTIONS = frozenset({"propose_accept", "propose_next", "propose_prev"})

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
        Binding("left_curly_bracket", "nudge_step_down", "step −", show=False, key_display="{"),
        Binding("right_curly_bracket", "nudge_step_up", "step +", show=False, key_display="}"),
        Binding("left_square_bracket", "speed_down", "slower", key_display="["),
        Binding("right_square_bracket", "speed_up", "faster", key_display="]"),
        Binding("i", "insert_chunk", "insert chunk"),
        Binding("I", "insert_labeled", "insert+label", show=False),
        Binding("x", "remove_insert", "remove insert", show=False),
        Binding("S", "split_chunk", "split chunk", show=False),
        Binding("e", "edit", "edit text"),
        Binding("y", "yank", "copy text"),
        Binding("right", "shift_push", "push word", key_display="→"),
        Binding("d", "shift_push", "push word", show=False),
        Binding("left", "shift_pull", "pull word", key_display="←"),
        Binding("a", "shift_pull", "pull word", show=False),
        Binding("tab", "cycle_lane", "lane", show=False, priority=True),
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
        Binding("m", "mark_quick", "mark"),
        Binding("b", "mark_boundary", "mark boundary"),
        Binding("M", "mark_editor", "mark+class"),
        Binding("n", "next_mark", "next mark"),
        Binding("N", "prev_mark", "prev mark"),
        Binding("p", "next_prune", "next prune"),
        Binding("P", "prev_prune", "prev prune"),
        Binding("F", "gate_editor", "flywheel gate", show=False),
        Binding("enter", "open_source", "open", show=False),
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
                 lane: Optional[str] = None,              # Starting pass lane ("walk" | "assign"); None = sidecar-persisted preference, else walk (DEC 8a4df244)
                 purpose: Optional[str] = None):          # None = genuine pass; "feature-test" tags the session excludable from flywheel datasets (--test, DEC c86714a4)
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
        self._input_mode = "edit"          # what the hidden Input commits ("edit" | "mark" | "insert" | "assign" | "split")
        self.lane = lane or "walk"         # active pass lane ("walk" | "assign"; tab cycles, sidecar persists)
        self._lane_arg = lane              # explicit --lane (wins over the sidecar; None = defer)
        self._entities: List[Dict[str, Any]] = []  # speaker Entity registry (graph-side, source-spanning)
        self._active_entity: Optional[str] = None  # entity id space assigns ("same speaker continues")
        self._accept_cluster: Optional[str] = None  # cluster awaiting its name (the a-gesture's editor hop; None = no accept pending)
        self._shift_busy = False           # in-flight boundary-shift commit (key-repeat throttle)
        self._last_shift = 0.0             # last completed shift (monotonic; paint-rate floor)
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
        self.lane = self._lane_arg or (saved_lane if saved_lane in ("walk", "assign") else "walk")
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
                chip = Text(f"?{p.get('label')} {float(p.get('score') or 0):.2f}{extra} ▏",
                            style="dim cyan")
                chip.append_text(body)
                body = chip
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
        badges = {"assign": "\\[ASSIGN]", "propose": "\\[PROPOSE]"}.get(self.lane, "\\[WALK]")
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
            return (f"{head}  ·  proposals {pending} pending"
                    f" · set {str(meta.get('proposal_set_id') or '')[-8:]}"
                    f" · model {str(meta.get('training_run_id') or '')[-8:]}{tail}"
                    f"  ·  a accept · n/N jump · ,./<> nudge · i/I manual · x remove"
                    f" · j/k walk · r replay · g/G seam · tab lane · q quit")
        edited = sum(1 for v in self._marks.values() if v == "corrected")
        return (f"{head}  ·  edited {edited}{tail}"
                f"  ·  j/k·w/s walk · ←→/a/d shift · r replay · g/G seam · ,./<> nudge"
                f" · {{}} step · \\[/] speed · e edit · y copy · i/I ⊕insert · x ⊖remove"
                f" · m/b/M ⚑mark · n/N⚑ p/P✂ jump · F gate · tab assign-lane · q quit")

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
        return action not in (self.ASSIGN_ONLY_ACTIONS | self.PROPOSE_ONLY_ACTIONS)

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

    def _play_cursor(self) -> None:
        seg = self.view.segments[self.cursor]
        if seg.id in self.view.inserted_ids:
            # Synthetic chunk: its audio may exist ONLY in the original source
            # (inter-chunk gaps by construction) — decode a source slice, the
            # seam-audition path. Zero/near-zero width has nothing to sound.
            self.player.stop()
            if seg.start_time is not None and seg.end_time is not None \
                    and float(seg.end_time) - float(seg.start_time) >= 0.02:
                self.run_worker(self._play_source_span(float(seg.start_time),
                                                       float(seg.end_time)))
            return
        c = self.view.chunk(self.cursor)
        if c is None:
            self.player.stop()
            return
        self.player.play(load_chunk(c.wav_path, c.start_s, c.end_s, speed=self.speed))

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
        if new == self.cursor:
            return
        self.cursor = new
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
        if self._input_mode == "gate":
            await self._submit_gate(event.value)
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
        # walk -> assign -> propose -> walk; the propose lane only enters the
        # rotation when a proposal set loaded (no set = the walk stays manual).
        order = ["walk", "assign"] + (["propose"] if self.view.proposals_meta else [])
        self.lane = order[(order.index(self.lane) + 1) % len(order)] \
            if self.lane in order else "walk"
        save_tui_state(self._graph_db_path, self.view.source_id, None, lane=self.lane)
        self._render()
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

    async def action_propose_accept(self) -> None:
        """a (propose lane): accept the cursor anchor's first pending proposal.

        The accept gesture IS the insert op (DEC 8e05b87b, the diarization-
        accept precedent): a labeled chunk-insert lands with the MODEL'S span;
        nudges on top are the edit record; skipping past is the unmarked
        reject — verdicts derive later from the bench join, never stored."""
        view = self.view
        seg = view.segments[self.cursor]
        props = view.event_proposals.get(seg.id)
        if not props:
            self.query_one("#status", Static).update(
                "no pending proposal at cursor — n/N jump to one")
            return
        p = props[0]
        plan = plan_chunk_insert(view.segments, self.cursor,
                                 inserted_ids=view.inserted_ids,
                                 insert_ranks=view.insert_ranks)
        if plan is None:
            self.query_one("#status", Static).update(
                "accept: refused (missing times, or an overlapping boundary "
                "— nudge the overlap first)")
            return
        start_s, end_s = float(p["start_time"]), float(p["end_time"])
        insert_id = await commit_chunk_insert_correction(
            view.queue, view.graph_id, view.source_id,
            plan["after_id"], start_s, end_s, self.session_id,
            before_segment_id=plan["before_id"], label=p.get("label"),
            rank=plan["rank"], actor=self.actor, journal_path=self._journal_path)
        pos = view.add_insert_local(
            {"id": insert_id,
             "payload": {"operation": "chunk_insert",
                         "after_segment_id": plan["after_id"],
                         "start_time": start_s, "end_time": end_s,
                         "label": p.get("label"), "text": "", "rank": plan["rank"]}})
        view.accept_proposal_local(seg.id, str(p.get("proposal_id")), insert_id)
        if pos is not None:
            self._marks = {(k + 1 if k >= pos else k): v for k, v in self._marks.items()}
            self.cursor = pos
        self._render()
        self.query_one("#status", Static).update(
            f"✓ accepted {p.get('label')} {start_s:.2f}–{end_s:.2f}s"
            f" (score {float(p.get('score') or 0):.2f}) · ,./<> nudge if off · playing span")
        self.player.stop()
        self.run_worker(self._play_source_span(start_s, end_s))

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
                self.query_one("#status", Static).update(
                    f"?{p.get('label')} {float(p['start_time']):.2f}–{float(p['end_time']):.2f}s"
                    f" score {float(p.get('score') or 0):.2f} · a accepts · playing span")
                self.player.stop()
                self.run_worker(self._play_source_span(
                    float(p["start_time"]), float(p["end_time"])))
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
        self._render()
        if info:
            status.update(f"⊖ unsplit: right half removed, target restored"
                          f" (text + end back to {seg.end_time:.2f}s)")
        else:
            status.update(f"⊖ removed inserted chunk"
                          f" ({seg.start_time:.2f}–{seg.end_time:.2f}s)")

    async def _play_source_span(self, start_s: float, end_s: float) -> None:
        """Decode + play a source-coordinate span of the ORIGINAL media — the
        inserted-chunk playback path (no model-input WAV need cover it)."""
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
        if view.segments[i].id in view.inserted_ids \
                or view.segments[i + 1].id in view.inserted_ids:
            status.update("boundary shift: ✋ inserted chunk — its text lives on the overlay (e edits it)")
            return
        if view.aseg_index(i) != view.aseg_index(i + 1):
            status.update("boundary shift: ✋ audio-segment seam — text stays within its audio segment")
            return
        left, right = view.segments[i], view.segments[i + 1]
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
        self._marks[i + 1] = "corrected"
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

    def action_cancel(self) -> None:
        editor = self.query_one("#editor", Input)
        if editor.display:
            self._accept_cluster = None  # an aborted a-gesture must not hijack the next assign
            self._close_editor()
            self._render()
        else:
            self.player.stop()

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
    nudge_step_ms: Optional[float] = None,  # Nudge-step preference (db-wide `_nudge_step_ms`; None = leave as-is)
    lane: Optional[str] = None,      # Pass-lane preference (db-wide `_lane`; None = leave as-is)
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
    if nudge_step_ms is not None:
        state["_nudge_step_ms"] = float(nudge_step_ms)
    if lane is not None:
        state["_lane"] = str(lane)
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
