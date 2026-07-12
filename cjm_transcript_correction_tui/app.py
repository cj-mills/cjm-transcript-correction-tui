from typing import Dict, List, Optional

from cjm_transcript_correction_core.graph import (commit_text_correction, record_review_markers,
                                                  start_session)
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, Static

from .audio import ChunkPlayer, load_chunk
from .spine import SpineView


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
        Binding("k", "prev", "prev"),
        Binding("up", "prev", "prev", show=False),
        Binding("r", "replay", "replay"),
        Binding("e", "edit", "edit text"),
        Binding("space", "reviewed", "mark reviewed"),
        Binding("escape", "cancel", "cancel/stop", show=False, priority=True),
        Binding("q", "quit_app", "quit"),
    ]

    def __init__(self, graph_db_path: str,                # The shared transcription graph db
                 *, source: Optional[str] = None,         # Source id or title substring
                 manifests_dir: str = ".cjm/manifests",   # Capability manifests directory
                 rendition: Optional[str] = None,         # Rendition selector (None = auto)
                 actor: str = "human",                    # Actor recorded on corrections
                 autoplay: bool = True):                  # Auto-play the focused chunk
        super().__init__()
        self._open_kwargs = dict(source=source, manifests_dir=manifests_dir,
                                 rendition=rendition)
        self._graph_db_path = graph_db_path
        self.view: Optional[SpineView] = None
        self.player: Optional[ChunkPlayer] = None
        self.cursor = 0
        self.actor = actor
        self.autoplay = autoplay
        self.session_id: Optional[str] = None
        self._marks: Dict[int, str] = {}   # cursor position -> local decision echo
        self._slots: List[Static] = []

    def compose(self) -> ComposeResult:
        yield Vertical(id="cards")
        yield Static("loading spine…", id="status")
        editor = Input(id="editor")
        editor.display = False
        yield editor

    async def on_mount(self) -> None:
        self.view = await SpineView.open(self._graph_db_path, **self._open_kwargs)
        self.player = ChunkPlayer()
        sess = await start_session(self.view.queue, self.view.graph_id,
                                   [self.view.source_id])
        self.session_id = sess.id
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
            t = (f"{seg.start_time:.1f}–{seg.end_time:.1f}s"
                 if seg.start_time is not None else "(no audio)")
            head = f"[bold]#{seg.index}[/bold]  {t}  {mark}"
            body = seg.text or "[dim](empty)[/dim]"
            text = f"{head}\n{body}"
            slot.update(f"[reverse]{text}[/reverse]" if pos == self.cursor else text)
        done = sum(1 for v in self._marks.values() if v)
        self.query_one("#status", Static).update(
            f"{view.source_title}  ·  segment {self.cursor + 1}/{view.size}"
            f"  ·  marked {done}  ·  session {str(self.session_id or '')[:8]}"
            f"  ·  j/k walk · r replay · e edit · space reviewed · q quit")

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

    def action_cancel(self) -> None:
        editor = self.query_one("#editor", Input)
        if editor.display:
            self._close_editor()
            self._render()
        else:
            self.player.stop()

    async def action_quit_app(self) -> None:
        if self.player is not None:
            self.player.close()
        if self.view is not None:
            await self.view.close()
        self.exit()
