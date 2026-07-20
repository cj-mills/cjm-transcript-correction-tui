from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cjm_context_graph_layer.grammar import OverlayRelations, SpineRelations
from cjm_context_graph_layer.ops import graph_task
from cjm_context_graph_primitives.query import NodeQuery, OrderBy, RelationPredicate
from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue
from cjm_transcript_correction_core.cli import load_capabilities
from cjm_transcript_correction_core.graph import (active_corrections, load_source_corrections,
                                                  load_source_segments, mark_anchor_segments,
                                                  open_marks, project_effective_spine,
                                                  resolve_source_renditions)
from cjm_transcript_correction_core.models import SpineSegment
from cjm_transcript_graph_schema.schema import TranscriptGraphLabels


@dataclass
class ChunkRef:
    """Where one Segment's VAD-chunk audio lives: the model-input WAV + the chunk-local span.

    The correction loop plays from the model-input rendition (what the model heard),
    so the span is expressed LOCAL to that AudioSegment's WAV — Segment times are
    source-coordinate on the graph; the join subtracts the owning AudioSegment's
    start."""
    wav_path: str   # The AudioSegment's model-input WAV (16 kHz mono)
    start_s: float  # Chunk start, seconds, local to `wav_path`
    end_s: float    # Chunk end, seconds, local to `wav_path`


class SpineView:
    """One Source's effective correction spine, cursor-windowed for the TUI.

    The driver's single seat at the graph: bootstraps the substrate capability stack
    (graph worker pointed at the shared transcription graph), resolves the rendition
    chain, loads the fine Segment spine, applies the effective projection (layer-0 +
    active corrections — the SAME read every downstream consumer gets), and joins
    each Segment to its model-input WAV chunk. The TUI renders WINDOWS of this view
    (`window(cursor, count)` — scrolling moves the CURSOR, slots re-bind around it;
    there is no viewport state) and slices audio via `chunk(i)`. Open MARKS load
    with the corrections (⚑ bookkeeping only — they never touch the projection).
    Reads come through correction-core's operation vocabulary; writes too — the
    TUI never touches the graph directly."""

    def __init__(self, manager: CapabilityManager, queue: JobQueue, graph_id: str,
                 source_id: str, source_title: str):
        self._manager = manager
        self.queue = queue
        self.graph_id = graph_id
        self.source_id = source_id
        self.source_title = source_title
        self.segments: List[SpineSegment] = []       # Full-skeleton effective spine (text edits applied, prunes marked)
        self.pruned_ids: set = set()                 # Segment ids a prune correction targets (card marks)
        self._prune_corrections: List[dict] = []     # Active prune Corrections (unprune anchors)
        self._open_marks: List[dict] = []            # OPEN mark Corrections (routed attention)
        self.marked_ids: set = set()                 # Segment ids an open mark anchors (⚑ glyphs)
        self.seen_mark_classes: List[str] = []       # DISTINCT classes journaled on this source (open or discharged)
        self._aseg_starts: List[float] = []          # AudioSegment starts (sorted, for bisect)
        self._aseg_audio: List[Optional[ChunkRef]] = []  # Parallel: (wav, aseg-start) join stubs

    @classmethod
    async def open(cls, graph_db_path: str,                 # The shared transcription graph db
                   *, source: Optional[str] = None,         # Source node id OR a title substring
                   manifests_dir: str = ".cjm/manifests",   # Capability manifests directory
                   graph_capability: str = "cjm-capability-graph-sqlite",
                   rendition: Optional[str] = None,         # Rendition selector (None = auto)
                   ) -> "SpineView":
        """Bootstrap the capability stack and load one Source's effective spine."""
        manager = CapabilityManager(search_paths=[Path(manifests_dir)])
        load_capabilities(manager, [graph_capability],
                          configs={graph_capability: {"db_path": str(graph_db_path)}})
        queue = JobQueue(deps=manager)
        await queue.start()
        try:
            sq = NodeQuery(label="Source", project=["title"])
            res = await graph_task(queue, graph_capability, "query_nodes", query=sq.to_dict())
            sources = [(r["id"], str(r.get("title") or "")) for r in (res.rows or [])]
            if source:
                picked = [(i, t) for i, t in sources
                          if i == source or source.lower() in t.lower()]
            else:
                picked = sources
            if len(picked) != 1:
                titles = "; ".join(t for _, t in sources)
                raise ValueError(f"need exactly one Source (matched {len(picked)}) — "
                                 f"pass `source=` an id or title substring; available: {titles}")
            view = cls(manager, queue, graph_capability, picked[0][0], picked[0][1])
            await view._load(rendition)
            return view
        except BaseException:
            await queue.stop()
            raise

    async def _load(self, rendition: Optional[str]) -> None:
        """Load spine + corrections + the audio join (one Source, one rendition chain)."""
        segments = await load_source_segments(self.queue, self.graph_id, self.source_id,
                                              rendition_selector=rendition)
        corrections, superseded = await load_source_corrections(
            self.queue, self.graph_id, self.source_id)
        active = active_corrections(corrections, superseded)
        # Open marks paint ⚑ in the walk; they NEVER touch the projection
        # (corrections_to_edits has no arm for correction_type "mark" — DEC 2a231843).
        self._open_marks = open_marks(corrections, superseded)
        self._recompute_marked_ids()
        # The correction surface walks the FULL VAD skeleton (the 1:1 invariant):
        # prune corrections are NOT applied to this view — an "empty" chunk may hold
        # speech that FA starved (the falsified D14 premise), and an empty chunk is
        # exactly where a boundary-shift pulls mis-assigned text back. Pruned ids
        # surface as card marks instead of disappearing from the walk.
        self._prune_corrections = [
            c for c in active
            if c.get("correction_type") == "grouping"
            and (c.get("payload") or {}).get("operation") == "prune_empty"]
        self.pruned_ids = {
            sid for c in self._prune_corrections
            for sid in (c.get("payload") or {}).get("pruned_segment_ids") or []}
        # Prunes are the ONLY corrections withheld from this view (their
        # positions stay walkable, marked); boundary shifts and text edits
        # APPLY, review verdicts map to no edit (58b2e0a0 residual fix).
        prune_ids = {c.get("id") for c in self._prune_corrections}
        projected = [c for c in active if c.get("id") not in prune_ids]
        self.segments = project_effective_spine(segments, projected)
        rend_ids = set(await resolve_source_renditions(
            self.queue, self.graph_id, self.source_id, rendition))
        aq = NodeQuery(label=TranscriptGraphLabels.AUDIO_SEGMENT,
                       related=RelationPredicate(SpineRelations.PART_OF, node_id=self.source_id),
                       order_by=OrderBy(prop="start"), project=["start", "end"])
        ares = await graph_task(self.queue, self.graph_id, "query_nodes", query=aq.to_dict())
        asegs = [(r["id"], float(r.get("start") or 0.0)) for r in (ares.rows or [])]
        rq = NodeQuery(label=TranscriptGraphLabels.AUDIO_RENDITION,
                       related=RelationPredicate(OverlayRelations.DERIVED_FROM,
                                                 node_ids=[a[0] for a in asegs]),
                       project=["model_input_path", "audio_segment_id"])
        rres = await graph_task(self.queue, self.graph_id, "query_nodes", query=rq.to_dict())
        wav_by_aseg: Dict[str, str] = {
            str(r.get("audio_segment_id")): str(r.get("model_input_path") or "")
            for r in (rres.rows or []) if r["id"] in rend_ids}
        self._aseg_starts = [start for _, start in asegs]
        self._aseg_audio = [
            ChunkRef(wav_by_aseg[aid], start, 0.0) if aid in wav_by_aseg else None
            for aid, start in asegs]

    @property
    def size(self) -> int:  # Total segments in the effective spine
        return len(self.segments)

    def window(self, cursor: int, count: int) -> List[SpineSegment]:
        """The slot window around the cursor — clamped, cursor-parameterized, stateless."""
        if not self.segments or count <= 0:
            return []
        half = count // 2
        start = max(0, min(max(0, cursor - half), len(self.segments) - count))
        return self.segments[start:start + count]

    def aseg_index(self, index: int) -> Optional[int]:
        """Which coarse AudioSegment (by position) a segment sits in — seam rendering."""
        if not (0 <= index < len(self.segments)) or not self._aseg_starts:
            return None
        seg = self.segments[index]
        if seg.start_time is None:
            return None
        return max(0, bisect_right(self._aseg_starts, float(seg.start_time)) - 1)

    def chunk(self, index: int) -> Optional[ChunkRef]:
        """The Segment's VAD-chunk audio ref (model-input WAV + chunk-local span), or None."""
        if not (0 <= index < len(self.segments)) or not self._aseg_starts:
            return None
        seg = self.segments[index]
        if seg.start_time is None or seg.end_time is None:
            return None
        i = max(0, bisect_right(self._aseg_starts, float(seg.start_time)) - 1)
        stub = self._aseg_audio[i]
        if stub is None or not stub.wav_path:
            return None
        return ChunkRef(stub.wav_path,
                        float(seg.start_time) - stub.start_s,
                        float(seg.end_time) - stub.start_s)

    def prune_correction_for(self, segment_id: str) -> Optional[dict]:
        """The active prune Correction covering a segment (the unprune anchor), or None."""
        for c in self._prune_corrections:
            if segment_id in ((c.get("payload") or {}).get("pruned_segment_ids") or []):
                return c
        return None

    def unprune_local(self, prior_id: str, amended: dict) -> None:
        """Local echo of a committed prune amendment (amended supersedes prior_id)."""
        self._prune_corrections = [amended if c.get("id") == prior_id else c
                                   for c in self._prune_corrections]
        self.pruned_ids = {
            sid for c in self._prune_corrections
            for sid in (c.get("payload") or {}).get("pruned_segment_ids") or []}

    def _recompute_marked_ids(self) -> None:
        """Re-derive the ⚑ id set + observed class list from the OPEN marks
        (load + local echoes). A class leaves the picker menu when its last
        open mark on this source is discharged — junk classes clean up via
        dismissal; proven classes persist by PROMOTION into the recommended
        slate, never by haunting the menu from discharged marks."""
        self.marked_ids = set()
        for m in self._open_marks:
            try:
                self.marked_ids.update(mark_anchor_segments(
                    (m.get("payload") or {}).get("anchor") or {}))
            except ValueError:
                continue   # malformed historical mark: skip its glyph, never break the walk
        self.seen_mark_classes = sorted({
            str((m.get("payload") or {}).get("mark_class"))
            for m in self._open_marks
            if (m.get("payload") or {}).get("mark_class")})

    def marks_for(self, segment_id: str) -> List[dict]:
        """The open marks anchored to a segment (oldest first) — dismissal targets."""
        out = []
        for m in self._open_marks:
            try:
                ids = mark_anchor_segments((m.get("payload") or {}).get("anchor") or {})
            except ValueError:
                continue
            if segment_id in ids:
                out.append(m)
        return out

    def add_mark_local(self, mark: dict) -> None:
        """Local echo of a committed mark (the ⚑ paints without a reload)."""
        self._open_marks.append(mark)
        self._recompute_marked_ids()

    def dismiss_mark_local(self, mark_id: str) -> None:
        """Local echo of a mark dismissal."""
        self._open_marks = [m for m in self._open_marks if m.get("id") != mark_id]
        self._recompute_marked_ids()

    async def close(self) -> None:
        """Tear down the queue + capability stack (app exit)."""
        await self.queue.stop()
        try:
            self._manager.unload_capability(self.graph_id)
        except Exception:
            pass


def plan_boundary_shift(
    left_text: str,   # The cursor segment's current effective text
    right_text: str,  # The next segment's current effective text
    direction: str,   # "push" (last word of left -> right) | "pull" (first word of right -> left)
) -> Optional[Tuple[str, str, str]]:  # (moved word, new left text, new right text); None = nothing to move
    """Plan a ONE-WORD boundary shift (the [ / ] gesture unit).

    Mirrors the layer's junction-normalizing semantics (DEC f83c6931) for the
    local echo: single-space joins, vacated boundary whitespace collapses.
    Repeat presses chain one-word corrections; the projection applies them in
    created_at order over the evolving text, so the chain composes exactly.
    """
    if direction == "push":
        words = left_text.split()
        if not words:
            return None
        moved = words[-1]
        base = left_text.rstrip()
        new_left = base[: len(base) - len(moved)].rstrip()
        rtext = right_text.lstrip()
        new_right = f"{moved} {rtext}" if rtext else moved
    else:
        words = right_text.split()
        if not words:
            return None
        moved = words[0]
        base = right_text.lstrip()
        new_right = base[len(moved):].lstrip()
        ltext = left_text.rstrip()
        new_left = f"{ltext} {moved}" if ltext else moved
    return moved, new_left, new_right


def parse_mark_input(
    raw: str,           # The mark-editor submission: `class ["snippet"] [note...]`
    segment_text: str,  # The focused segment's current effective text (span lookup)
) -> Optional[Tuple[str, Optional[Tuple[int, int, str]], Optional[str]]]:  # (class, span, note); None = empty input
    """Parse the M-editor mark grammar (pure; the DEC 2a231843 TUI gesture).

    First token = the mark class (open vocabulary). An optional "double-quoted"
    snippet that occurs in the segment text becomes a SPAN anchor (first
    occurrence; offsets + verbatim snapshot). Everything else is the note.
    A quoted snippet NOT found in the text stays part of the note — the mark
    degrades to segment scope rather than recording a false span.
    """
    text = (raw or "").strip()
    if not text:
        return None
    head, _, rest = text.partition(" ")
    rest = rest.strip()
    span = None
    if rest.startswith('"') and '"' in rest[1:]:
        snippet, _, tail = rest[1:].partition('"')
        at = segment_text.find(snippet) if snippet else -1
        if at != -1:
            span = (at, at + len(snippet), snippet)
            rest = tail.strip()
    return head, span, (rest or None)


def resolve_mark_class_token(
    raw: str,         # The mark-editor submission (possibly `N ...`)
    menu: List[str],  # Selectable classes (recommended slate + observed)
) -> Tuple[str, Optional[str]]:  # (possibly-rewritten submission, error message or None)
    """Resolve a leading digit token to its menu class (the M picker; pure).

    `2 "snippet" note` becomes `<menu[1]> "snippet" note` — everything after
    the digit is preserved VERBATIM (a snippet's inner spacing must survive).
    Out-of-range numbers return an error instead of minting a numeric class.
    """
    head, _, rest = (raw or "").strip().partition(" ")
    if not head.isdigit():
        return raw, None
    n = int(head)
    if not (1 <= n <= len(menu)):
        return raw, f"no class #{n} (menu is 1-{len(menu)})"
    return f"{menu[n - 1]} {rest}".strip(), None
