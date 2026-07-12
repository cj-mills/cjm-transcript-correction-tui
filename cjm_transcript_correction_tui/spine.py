from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from cjm_context_graph_layer.grammar import OverlayRelations, SpineRelations
from cjm_context_graph_layer.ops import graph_task
from cjm_context_graph_primitives.query import NodeQuery, OrderBy, RelationPredicate
from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue
from cjm_transcript_correction_core.cli import load_capabilities
from cjm_transcript_correction_core.graph import (active_corrections, load_source_corrections,
                                                  load_source_segments, project_effective_spine,
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
    there is no viewport state) and slices audio via `chunk(i)`. Reads come through
    correction-core's operation vocabulary; writes will too — the TUI never touches
    the graph directly."""

    def __init__(self, manager: CapabilityManager, queue: JobQueue, graph_id: str,
                 source_id: str, source_title: str):
        self._manager = manager
        self._queue = queue
        self.graph_id = graph_id
        self.source_id = source_id
        self.source_title = source_title
        self.segments: List[SpineSegment] = []       # Effective spine, index order
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
        segments = await load_source_segments(self._queue, self.graph_id, self.source_id,
                                              rendition_selector=rendition)
        corrections, superseded = await load_source_corrections(
            self._queue, self.graph_id, self.source_id)
        self.segments = project_effective_spine(
            segments, active_corrections(corrections, superseded))
        rend_ids = set(await resolve_source_renditions(
            self._queue, self.graph_id, self.source_id, rendition))
        aq = NodeQuery(label=TranscriptGraphLabels.AUDIO_SEGMENT,
                       related=RelationPredicate(SpineRelations.PART_OF, node_id=self.source_id),
                       order_by=OrderBy(prop="start"), project=["start", "end"])
        ares = await graph_task(self._queue, self.graph_id, "query_nodes", query=aq.to_dict())
        asegs = [(r["id"], float(r.get("start") or 0.0)) for r in (ares.rows or [])]
        rq = NodeQuery(label=TranscriptGraphLabels.AUDIO_RENDITION,
                       related=RelationPredicate(OverlayRelations.DERIVED_FROM,
                                                 node_ids=[a[0] for a in asegs]),
                       project=["model_input_path", "audio_segment_id"])
        rres = await graph_task(self._queue, self.graph_id, "query_nodes", query=rq.to_dict())
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

    async def close(self) -> None:
        """Tear down the queue + capability stack (app exit)."""
        await self._queue.stop()
        try:
            self._manager.unload_capability(self.graph_id)
        except Exception:
            pass
