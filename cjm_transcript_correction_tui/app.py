import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cjm_transcript_correction_core.graph import (commit_boundary_shift_correction,
                                                  commit_prune_amendment, commit_text_correction,
                                                  record_review_markers, start_session)
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, Static

from .audio import ChunkPlayer, load_chunk
from .spine import plan_boundary_shift, SpineView


class CorrectionApp(App):
    """The correction loop, v0 thinnest slice: document-order segment walk with
    VAD-chunk auto-play and in-place fidelity edits, over the shared transcription
    graph through correction-core's operation vocabulary.

    Interaction contract (DEC 54640079 + the walkthrough capture): the surface is a
    fixed-slot window over the cursor-parameterized effective spine — scrolling
    (keys AND wheel) moves the CURSOR, slots re-bind around it, nothing moves
    unbidden, content never overlaps. Focusing a segment auto-plays its VAD chunk
    from the model-input WAV (immediate-play; churn accepted per the spike). An
    edit commits a `text_content` Correction (+ its REVIEWED marker) and updates
    the local effective text — decisions persist, the worklist stays derived.
    The graph stack opens INSIDE the app (`on_mount`) so the JobQueue lives on
    Textual's event loop."""

    AUTO_FOCUS = None  # the hidden editor Input must not swallow the walk keys at mount

    BINDINGS = [
        Binding("j", "next", "next"),
        Binding("down", "next", "next", show=False),
        Binding("s", "next", "next", show=False),
        Binding("k", "prev", "prev"),
        Binding("up", "prev", "prev", show=False),
        Binding("w", "prev", "prev", show=False),
        Binding("r", "replay", "replay"),
        Binding("e", "edit", "edit text"),
        Binding("right", "shift_push", "push word", key_display="→"),
        Binding("d", "shift_push", "push word", show=False),
        Binding("left", "shift_pull", "pull word", key_display="←"),
        Binding("a", "shift_pull", "pull word", show=False),
        Binding("space", "reviewed", "mark reviewed"),
        Binding("escape", "cancel", "cancel/stop", show=False, priority=True),
        Binding("q", "quit_app", "quit"),
    ]

    def __init__(self, graph_db_path: str,                # The shared transcription graph db
                 *, source: Optional[str] = None,         # Source id or title substring
                 manifests_dir: str = ".cjm/manifests",   # Capability manifests directory
                 rendition: Optional[str] = None,         # Rendition selector (None = auto)
                 actor: str = "human",                    # Actor recorded on corrections
                 autoplay: bool = True,                   # Auto-play the focused chunk
                 audio_device: Optional[object] = None,   # Output device (None = system default)
                 resume: bool = True,                     # Reopen at the source's last-focused segment
                 shift_floor_s: float = 0.08):            # Min seconds between held-key boundary shifts
        super().__init__()
        self._open_kwargs = dict(source=source, manifests_dir=manifests_dir,
                                 rendition=rendition)
        self._graph_db_path = graph_db_path
        self.view: Optional[SpineView] = None
        self.player: Optional[ChunkPlayer] = None
        self.cursor = 0
        self.actor = actor
        self.autoplay = autoplay
        self.audio_device = audio_device
        self.session_id: Optional[str] = None
        self._marks: Dict[int, str] = {}   # cursor position -> local decision echo
        self._shift_busy = False           # in-flight boundary-shift commit (key-repeat throttle)
        self._last_shift = 0.0             # last completed shift (monotonic; paint-rate floor)
        self._shift_floor = float(shift_floor_s)  # tune with tests_manual/keyrate_probe.py
        self.resume = resume
        self._state_saved = 0.0            # last sidecar bookmark write (monotonic; 1s throttle)
        self._slots: List[Static] = []

    def compose(self) -> ComposeResult:
        yield Vertical(id="cards")
        yield Static("loading spine…", id="status")
        editor = Input(id="editor")
        editor.display = False
        yield editor

    async def on_mount(self) -> None:
        self.view = await SpineView.open(self._graph_db_path, **self._open_kwargs)
        self.player = ChunkPlayer(device=self.audio_device)
        sess = await start_session(self.view.queue, self.view.graph_id,
                                   [self.view.source_id])
        self.session_id = sess.id
        if self.resume:
            saved = load_tui_state(self._graph_db_path).get(self.view.source_id)
            if saved and self.view.size:
                self.cursor = max(0, min(self.view.size - 1, int(saved.get("cursor", 0))))
        await self._build_slots()
        self._render()
        if self.autoplay:
            self._play_cursor()

    async def on_resize(self, event) -> None:
        if self.view is not None:
            await self._build_slots()
            self._render()

    async def _build_slots(self) -> None:
        """Fixed slot count sized to the terminal (the FastHTML-era slot model):
        each card budget ~4 lines + chrome; slots re-bind, they are never scrolled."""
        cards = self.query_one("#cards", Vertical)
        await cards.remove_children()
        n = max(3, (self.size.height - 4) // 4)
        self._slots = [Static("", classes="card") for _ in range(n)]
        for s in self._slots:
            await cards.mount(s)

    def _render(self) -> None:
        view, n = self.view, len(self._slots)
        window = view.window(self.cursor, n)
        half = n // 2
        start = max(0, min(max(0, self.cursor - half), max(0, view.size - n)))
        for slot, seg in zip(self._slots, window + [None] * (n - len(window))):
            if seg is None:
                slot.update("")
                continue
            pos = start + window.index(seg)
            mark = {"reviewed": "✓", "corrected": "✎"}.get(self._marks.get(pos, ""), "·")
            if seg.id in view.pruned_ids:
                mark += "  [red]✂ pruned-empty[/red]"
            t = (f"{seg.start_time:.1f}–{seg.end_time:.1f}s"
                 if seg.start_time is not None else "(no audio)")
            head = f"[bold]#{seg.index}[/bold]  {t}  {mark}"
            a = view.aseg_index(pos)
            if a is not None and (pos == 0 or view.aseg_index(pos - 1) != a):
                head = f"[yellow]━━━ audio segment {a} ━━━[/yellow]\n{head}"
            body = seg.text or "[dim](empty)[/dim]"
            text = f"{head}\n{body}"
            slot.update(f"[reverse]{text}[/reverse]" if pos == self.cursor else text)
        done = sum(1 for v in self._marks.values() if v)
        self.query_one("#status", Static).update(
            f"{view.source_title}  ·  segment {self.cursor + 1}/{view.size}"
            f"  ·  marked {done}  ·  session {str(self.session_id or '')[:8]}"
            f"  ·  j/k·w/s walk · ←→/a/d shift · r replay · e edit · space reviewed · q quit")

    def _play_cursor(self) -> None:
        c = self.view.chunk(self.cursor)
        if c is None:
            self.player.stop()
            return
        self.player.play(load_chunk(c.wav_path, c.start_s, c.end_s))

    def _move(self, delta: int) -> None:
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

    def action_edit(self) -> None:
        editor = self.query_one("#editor", Input)
        editor.value = self.view.segments[self.cursor].text
        editor.display = True
        editor.focus()

    async def on_input_submitted(self, event) -> None:
        seg = self.view.segments[self.cursor]
        new_text = event.value
        if new_text != seg.text:
            await commit_text_correction(
                self.view.queue, self.view.graph_id, self.view.source_id,
                seg.id, new_text, self.session_id,
                old_text=seg.text, actor=self.actor)
            seg.text = new_text          # local echo of the new effective text
            self._marks[self.cursor] = "corrected"
        self._close_editor()
        self._render()

    def _close_editor(self) -> None:
        editor = self.query_one("#editor", Input)
        editor.display = False
        self.set_focus(None)

    async def action_reviewed(self) -> None:
        seg = self.view.segments[self.cursor]
        await record_review_markers(self.view.queue, self.view.graph_id,
                                    self.session_id, [(seg.id, "reviewed")])
        self._marks.setdefault(self.cursor, "reviewed")
        self._move(1)

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
        left, right = view.segments[i], view.segments[i + 1]
        plan = plan_boundary_shift(left.text, right.text, direction)
        if plan is None:
            status.update(f"boundary shift: nothing to {direction}")
            return
        moved, new_left, new_right = plan
        await commit_boundary_shift_correction(
            view.queue, view.graph_id, view.source_id, left.id, right.id,
            moved, direction, self.session_id, actor=self.actor)
        receiver = right if direction == "push" else left
        if receiver.id in view.pruned_ids:
            prior = view.prune_correction_for(receiver.id)
            if prior is not None:
                amended = await commit_prune_amendment(
                    view.queue, view.graph_id, prior, [receiver.id],
                    self.session_id, actor=self.actor)
                view.unprune_local(prior["id"], amended)
        left.text, right.text = new_left, new_right   # local echo (same math as the layer)
        self._marks[i] = "corrected"
        self._marks[i + 1] = "corrected"
        self._render()

    async def action_shift_push(self) -> None:
        await self._shift_boundary("push")

    async def action_shift_pull(self) -> None:
        await self._shift_boundary("pull")

    def action_cancel(self) -> None:
        editor = self.query_one("#editor", Input)
        if editor.display:
            self._close_editor()
            self._render()
        else:
            self.player.stop()

    async def action_quit_app(self) -> None:
        if self.view is not None:
            save_tui_state(self._graph_db_path, self.view.source_id, self.cursor)
        if self.player is not None:
            self.player.close()
        if self.view is not None:
            await self.view.close()
        self.exit()


def load_tui_state(
    graph_db_path: str,  # The graph db whose sidecar state file to read
) -> Dict[str, Any]:  # {source_id: {"cursor": int, "ts": float}}; empty when absent/corrupt
    """Read the per-graph TUI sidecar state (last-focused positions)."""
    try:
        return json.loads(Path(f"{graph_db_path}.tui-state.json").read_text())
    except (OSError, ValueError):
        return {}


def save_tui_state(
    graph_db_path: str,  # The graph db whose sidecar state file to write
    source_id: str,      # Source whose position is being remembered
    cursor: int,         # Last-focused segment position
) -> None:
    """Merge one source's last-focused position into the sidecar state file.

    VIEW state, not knowledge — it lives in a local sidecar next to the db,
    never as a graph write (the cursor is where the eye was, not a decision).
    Write failures are silently tolerated: losing a bookmark must never break
    the correction loop."""
    state = load_tui_state(graph_db_path)
    state[source_id] = {"cursor": int(cursor), "ts": time.time()}
    try:
        Path(f"{graph_db_path}.tui-state.json").write_text(json.dumps(state, indent=1))
    except OSError:
        pass
