"""Tests for the [ / ] boundary-shift gesture planning + unprune bookkeeping.

Hermetic: plan_boundary_shift is pure; the SpineView prune bookkeeping is
exercised on a directly-constructed view (no graph stack)."""
from cjm_transcript_correction_tui.spine import parse_mark_input, resolve_mark_class_token, SpineView, plan_boundary_shift


def test_plan_push_and_pull_stripped_corpus():
    # push: last word of the cursor segment crosses right, single-space junction
    assert plan_boundary_shift("Mr. Gorbachev, tear", "down this wall.", "push") == (
        "tear", "Mr. Gorbachev,", "tear down this wall.")
    # pull: first word of the next segment crosses left
    assert plan_boundary_shift("Mr. Gorbachev,", "tear down this wall.", "pull") == (
        "tear", "Mr. Gorbachev, tear", "down this wall.")


def test_plan_empty_neighbor_and_noops():
    # the falsified-D14 rescue shape: push into a starved (empty) chunk
    assert plan_boundary_shift("largest naval battle in history", "", "push") == (
        "history", "largest naval battle in", "history")
    # nothing to give -> no-op
    assert plan_boundary_shift("", "down this wall.", "push") is None
    assert plan_boundary_shift("Mr. Gorbachev,", "", "pull") is None
    # whitespace-only counts as nothing
    assert plan_boundary_shift("   ", "x", "push") is None


def test_plan_chain_composes():
    # two pushes = the last two words move, one press at a time
    p1 = plan_boundary_shift("a b c", "d", "push")
    assert p1 == ("c", "a b", "c d")
    p2 = plan_boundary_shift(p1[1], p1[2], "push")
    assert p2 == ("b", "a", "b c d")
    # a pull undoes the last push
    p3 = plan_boundary_shift(p2[1], p2[2], "pull")
    assert p3 == ("b", "a b", "c d")


def test_unprune_bookkeeping():
    view = SpineView(manager=None, queue=None, graph_id="g", source_id="s",
                     source_title="t")
    prune = {"id": "p1", "correction_type": "grouping",
             "payload": {"operation": "prune_empty", "source_id": "s",
                         "pruned_segment_ids": ["b", "e"]}}
    view._prune_corrections = [prune]
    view.pruned_ids = {"b", "e"}
    assert view.prune_correction_for("b") is prune
    assert view.prune_correction_for("a") is None

    amended = {"id": "p2", "correction_type": "grouping",
               "payload": {"operation": "prune_empty", "source_id": "s",
                           "pruned_segment_ids": ["e"]}}
    view.unprune_local("p1", amended)
    assert view.pruned_ids == {"e"}
    assert view.prune_correction_for("b") is None
    assert view.prune_correction_for("e") is amended


def test_parse_mark_input_grammar():
    text = "Steve Jobs and Wozniak where like"
    # class only (the quick default shape)
    assert parse_mark_input("suspect", text) == ("suspect", None, None)
    # class + found snippet -> span (first occurrence, verbatim snapshot)
    assert parse_mark_input('homophone-substitution "where"', text) == (
        "homophone-substitution", (23, 28, "where"), None)
    # snippet + note
    assert parse_mark_input('homophone-substitution "where" context favors were', text) == (
        "homophone-substitution", (23, 28, "where"), "context favors were")
    # class + note (no snippet)
    assert parse_mark_input("repeat-omission dropped a repeated word", text) == (
        "repeat-omission", None, "dropped a repeated word")
    # snippet NOT in the text: degrade to segment scope, quotes stay in the note
    assert parse_mark_input('suspect "nowhere here" hm', text) == (
        "suspect", None, '"nowhere here" hm')
    # empty input = cancel
    assert parse_mark_input("   ", text) is None


def test_spineview_mark_bookkeeping():
    # hermetic: bookkeeping only, no graph stack
    view = SpineView.__new__(SpineView)
    view._open_marks = []
    view.marked_ids = set()
    view.seen_mark_classes = []
    mark = {"id": "m1", "correction_type": "mark",
            "payload": {"operation": "mark", "mark_class": "suspect",
                        "anchor": {"kind": "boundary", "boundary_after": "a",
                                   "right_segment_id": "b"}}}
    view.add_mark_local(mark)
    assert view.marked_ids == {"a", "b"}
    assert view.seen_mark_classes == ["suspect"]   # freshly minted class joins the menu
    assert [m["id"] for m in view.marks_for("a")] == ["m1"]
    # malformed historical marks never break the walk
    view.add_mark_local({"id": "m2", "payload": {"anchor": {"kind": "nope"}}})
    assert view.marked_ids == {"a", "b"}
    view.dismiss_mark_local("m1")
    assert view.marks_for("b") == [] and view.marked_ids == set()
    assert view.seen_mark_classes == []   # a class leaves the menu with its last open mark


def test_resolve_mark_class_token():
    menu = ["hesitation-omission", "repeat-omission", "foreign-speech"]
    # leading digit picks from the menu; the rest survives verbatim
    assert resolve_mark_class_token("2 note here", menu) == ("repeat-omission note here", None)
    assert resolve_mark_class_token('3 "a  spaced snippet" x', menu) == (
        'foreign-speech "a  spaced snippet" x', None)
    assert resolve_mark_class_token("2", menu) == ("repeat-omission", None)
    # explicit class names pass through untouched
    assert resolve_mark_class_token("repeat-omission x", menu) == ("repeat-omission x", None)
    # out-of-range numbers error instead of minting a numeric class
    raw, err = resolve_mark_class_token("9", menu)
    assert raw == "9" and err is not None
    raw, err = resolve_mark_class_token("0 note", menu)
    assert raw == "0 note" and err is not None


def test_match_sources_selector_arms():
    """2ce81638 discovery: the --source selector is pure and shared by direct
    open and the picker seed — exact-id wins, title substring is case-blind,
    None selects all, a miss selects none (the app widens a miss to the full
    picker instead of dead-ending)."""
    from cjm_transcript_correction_tui.spine import match_sources
    sources = [("id-a", "Intro꞉ Learning Games"), ("id-b", "Chapter One")]
    assert match_sources(sources, None) == sources
    assert match_sources(sources, "id-a") == [sources[0]]
    assert match_sources(sources, "chapter") == [sources[1]]
    assert match_sources(sources, "LEARNING") == [sources[0]]
    assert match_sources(sources, "zzz") == []


def test_seam_span_math():
    """g/G seam resolution (6beaa0e4, fine-boundary semantics — the first-drive
    correction): the boundary is between the CURSOR SEGMENT and its neighbor,
    context clamps to each segment's extent, the whole gap is covered, the gap
    stays signed (negative = overlap), None past the spine edges."""
    from types import SimpleNamespace

    def seg(start, end):
        return SimpleNamespace(start_time=start, end_time=end)

    view = SpineView.__new__(SpineView)
    view.segments = [seg(0.0, 5.0), seg(10.0, 28.0), seg(30.5, 40.0)]
    ref = view.seam(0, 1)   # boundary #0|#1: 2s context each side of the 5s gap
    assert (ref.left, ref.right) == (0, 1)
    assert (ref.start_s, ref.end_s) == (3.0, 12.0)
    assert ref.gap_s == 5.0
    # G from the next segment reaches the SAME boundary
    back = view.seam(1, -1)
    assert (back.start_s, back.end_s) == (ref.start_s, ref.end_s)
    # boundary #1|#2 (2.5s gap), reached backward from the last segment
    ref2 = view.seam(2, -1)
    assert (ref2.start_s, ref2.end_s) == (26.0, 32.5)
    assert abs(ref2.gap_s - 2.5) < 1e-9
    # context clamps to a segment shorter than the margin
    view.segments[0] = seg(4.0, 5.0)
    assert view.seam(0, 1).start_s == 4.0
    # contiguous sentence cuts inside one VAD chunk: zero gap, still auditable
    view.segments[1] = seg(5.0, 10.0)
    tight = view.seam(0, 1)
    assert tight.gap_s == 0.0 and (tight.start_s, tight.end_s) == (4.0, 7.0)
    # OVERLAPPING neighbors (the 2e42a737 timing-defect class): signed gap
    view.segments[1] = seg(4.5, 10.0)
    assert view.seam(0, 1).gap_s == -0.5
    # edges: nothing before the first / after the last segment
    assert view.seam(0, -1) is None
    assert view.seam(2, 1) is None
    # a neighbor without audio times resolves no seam
    view.segments.append(SimpleNamespace(start_time=None, end_time=None))
    assert view.seam(3, 1) is None and view.seam(2, 1) is None


def test_load_source_slice_decodes_original_media(tmp_path):
    """The seam decode goes back to the ORIGINAL media via ffmpeg: the slice's
    sample count matches the span and real audio comes out; a missing file
    fails loudly (skips on ffmpeg-less hosts)."""
    import asyncio
    import shutil

    import numpy as np
    import pytest
    import soundfile as sf

    from cjm_transcript_correction_tui.spine import load_source_slice
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    wav = tmp_path / "src.wav"
    sf.write(str(wav), np.sin(2 * np.pi * 440.0 * t) * 0.5, sr)
    samples = asyncio.run(load_source_slice(str(wav), 0.5, 1.5))
    assert samples.dtype == np.float32
    assert abs(len(samples) - sr) <= sr // 100   # ~1.0 s of audio
    assert float(np.abs(samples).max()) > 0.1    # sound, not silence
    with pytest.raises(RuntimeError):
        asyncio.run(load_source_slice(str(tmp_path / "gone.wav"), 0.0, 1.0))


def test_plan_time_nudge_welds_and_refusals():
    """3f9948d6 planner: welded point cuts move BOTH edges in one plan (the
    2e42a737 Example-A class), gapped boundaries move exactly one edge (each
    key pair reverses itself), collapse and missing-times cases refuse."""
    from types import SimpleNamespace

    from cjm_transcript_correction_tui.spine import plan_time_nudge

    def seg(sid, start, end):
        return SimpleNamespace(id=sid, start_time=start, end_time=end)

    # welded sentence cut at 5.0 (exact shared boundary)
    welded = [seg("a", 0.0, 5.0), seg("b", 5.0, 9.0)]
    plan = plan_time_nudge(welded, 0, "end", 0.1)
    assert [(e["segment_id"], e["edge"], e["new_time"]) for e in plan] == [
        ("a", "end", 5.1), ("b", "start", 5.1)]
    # same cut from the right seat, via the start pair — the mirror weld
    plan = plan_time_nudge(welded, 1, "start", -0.1)
    assert [(e["segment_id"], e["edge"]) for e in plan] == [("b", "start"), ("a", "end")]
    assert all(abs(e["new_time"] - 4.9) < 1e-9 for e in plan)

    # gapped boundary: ONE edge moves, the neighbor stays untouched
    gapped = [seg("a", 0.0, 5.0), seg("b", 6.0, 9.0)]
    plan = plan_time_nudge(gapped, 0, "end", 0.1)
    assert len(plan) == 1 and plan[0]["segment_id"] == "a"
    plan = plan_time_nudge(gapped, 1, "start", -0.1)
    assert len(plan) == 1 and plan[0]["segment_id"] == "b"

    # collapse refusals: an edge may not cross its own segment's other edge
    tiny = [seg("a", 0.0, 0.05), seg("b", 0.05, 9.0)]
    assert plan_time_nudge(tiny, 0, "end", -0.1) is None
    # welded collapse: the nudge would erase the NEIGHBOR
    sliver = [seg("a", 0.0, 5.0), seg("b", 5.0, 5.05)]
    assert plan_time_nudge(sliver, 0, "end", 0.1) is None
    # missing times / bad edge / out of range
    assert plan_time_nudge([seg("a", None, None)], 0, "end", 0.1) is None
    assert plan_time_nudge(welded, 0, "middle", 0.1) is None
    assert plan_time_nudge(welded, 5, "end", 0.1) is None
    # start of the spine cannot go negative
    assert plan_time_nudge([seg("a", 0.0, 5.0)], 0, "start", -0.1) is None
