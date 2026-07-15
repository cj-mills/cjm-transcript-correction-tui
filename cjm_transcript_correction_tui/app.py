import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_transcript_correction_core.graph import (commit_boundary_shift_correction,
                                                  commit_prune_amendment, commit_text_correction,
                                                  record_review_markers, start_session)
from cjm_transcript_correction_core.journal import sidecar_journal_path
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, Static

from .audio import ChunkPlayer, load_chunk
from .spine import plan_boundary_shift, SpineView


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
        Binding("left_square_bracket", "speed_down", "slower", key_display="["),
        Binding("right_square_bracket", "speed_up", "faster", key_display="]"),
        Binding("e", "edit", "edit text"),
        Binding("y", "yank", "copy text"),
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
                 shift_floor_s: float = 0.0):             # Min seconds between held-key boundary shifts (0 = ungoverned; the commit guard is the real governor)
        super().__init__()
        self._open_kwargs = dict(source=source, manifests_dir=manifests_dir,
                                 rendition=rendition)
        self._graph_db_path = graph_db_path
        # Every correction write appends through to the db's sidecar journal (DEC ccbab9f5).
        self._journal_path = sidecar_journal_path(graph_db_path)
        self.view: Optional[SpineView] = None
        self.player: Optional[ChunkPlayer] = None
        self.cursor = 0
        self.actor = actor
        self.autoplay = autoplay
        self.speed = 1.0                   # playback rate ([ ] preset ladder; sidecar-persisted preference)
        self.audio_device = audio_device
        self.session_id: Optional[str] = None
        self._marks: Dict[int, str] = {}   # cursor position -> local decision echo
        self._shift_busy = False           # in-flight boundary-shift commit (key-repeat throttle)
        self._last_shift = 0.0             # last completed shift (monotonic; paint-rate floor)
        self._shift_floor = float(shift_floor_s)  # tune with tests_manual/keyrate_probe.py
        self.resume = resume
        self._state_saved = 0.0            # last sidecar bookmark write (monotonic; 1s throttle)

    def compose(self) -> ComposeResult:
        yield Static("", id="cards")
        yield Static("loading spine…", id="status")
        editor = Input(id="editor")
        editor.display = False
        yield editor

    async def on_mount(self) -> None:
        self.view = await SpineView.open(self._graph_db_path, **self._open_kwargs)
        self.player = ChunkPlayer(device=self.audio_device)
        sess = await start_session(self.view.queue, self.view.graph_id,
                                   [self.view.source_id],
                                   journal_path=self._journal_path)
        self.session_id = sess.id
        state = load_tui_state(self._graph_db_path)
        try:
            # Speed is a PREFERENCE, not a position — restored even with resume=False.
            self.speed = float(state.get("_speed") or 1.0)
        except (TypeError, ValueError):
            self.speed = 1.0
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
        mark = {"reviewed": "✓", "corrected": "✎"}.get(self._marks.get(pos, ""), "·")
        # Gutter styling must ride SPANS, not the Text base style: lane text is
        # appended onto these same row objects, and a base style would bleed
        # into it (the round-2 drive regression — first two lane lines dimmed).
        g1 = Text()
        g1.append(f"#{seg.index} {mark}", style="dim")
        if seg.id in view.pruned_ids:
            g1.append(" ✂", style="red")
        g2 = Text()
        g2.append(f"{seg.start_time:.1f}–{seg.end_time:.1f}s"
                  if seg.start_time is not None else "(no audio)", style="dim")
        body = Text(seg.text) if seg.text else Text("(empty)", style="dim")
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
        return max(t_w, len("(no audio)"), len(f"#{last.index}") + 4) + 2

    def _render(self) -> None:
        """Center-pinned paint (drive round 4): the focused card's FIRST TEXT LINE
        is pinned to the vertical center of the card area; neighbor cards stack
        outward from it (one blank separator row) and absorb the height variance,
        clipping at the screen edges. The pin never moves — the spine flows past it."""
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
        done = sum(1 for v in self._marks.values() if v)
        self.query_one("#status", Static).update(
            f"{view.source_title}  ·  segment {self.cursor + 1}/{view.size}"
            f"  ·  marked {done}  ·  ×{self.speed:g}  ·  session {str(self.session_id or '')[:8]}"
            f"  ·  j/k·w/s walk · ←→/a/d shift · r replay · \\[/] speed · e edit · y copy"
            f" · space reviewed · q quit")

    def _play_cursor(self) -> None:
        c = self.view.chunk(self.cursor)
        if c is None:
            self.player.stop()
            return
        self.player.play(load_chunk(c.wav_path, c.start_s, c.end_s, speed=self.speed))

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
                old_text=seg.text, actor=self.actor,
                journal_path=self._journal_path)
            seg.text = new_text          # local echo of the new effective text
            self._marks[self.cursor] = "corrected"
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

    async def action_reviewed(self) -> None:
        seg = self.view.segments[self.cursor]
        await record_review_markers(self.view.queue, self.view.graph_id,
                                    self.session_id, [(seg.id, "reviewed")],
                                    journal_path=self._journal_path)
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

    def action_cancel(self) -> None:
        editor = self.query_one("#editor", Input)
        if editor.display:
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
    speed: Optional[float] = None,  # Playback-rate preference (db-wide `_speed`; None = leave as-is)
) -> None:
    """Merge one source's last-focused position into the sidecar state file.

    VIEW state, not knowledge — it lives in a local sidecar next to the db,
    never as a graph write (the cursor is where the eye was, not a decision).
    Write failures are silently tolerated: losing a bookmark must never break
    the correction loop."""
    state = load_tui_state(graph_db_path)
    state[source_id] = {"cursor": int(cursor), "ts": time.time()}
    if speed is not None:
        state["_speed"] = float(speed)
    try:
        Path(f"{graph_db_path}.tui-state.json").write_text(json.dumps(state, indent=1))
    except OSError:
        pass
