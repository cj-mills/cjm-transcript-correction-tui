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


def test_plan_chunk_insert_gap_weld_tail_refusals():
    """DEC 3d3fa2a8 gesture unit: whole-gap insert by default (the de994164
    missed-dispatch case is one keystroke), ZERO-WIDTH at welded cuts and at
    the spine tail (grown by the nudge keys), refusals for overlaps beyond the
    weld eps and missing times."""
    from types import SimpleNamespace

    from cjm_transcript_correction_tui.spine import plan_chunk_insert

    def seg(sid, start, end):
        return SimpleNamespace(id=sid, start_time=start, end_time=end)

    # real gap: the whole span, one keystroke
    gapped = [seg("a", 0.0, 4.5), seg("b", 6.0, 9.0)]
    plan = plan_chunk_insert(gapped, 0)
    assert plan == {"after_id": "a", "before_id": "b",
                    "start_s": 4.5, "end_s": 6.0, "welded": False, "rank": 0.0}

    # welded point cut: zero-width at the cut, nudges grow it over the bookends
    welded = [seg("a", 0.0, 5.0), seg("b", 5.0, 9.0)]
    plan = plan_chunk_insert(welded, 0)
    assert plan["welded"] and plan["start_s"] == plan["end_s"] == 5.0
    assert plan["after_id"] == "a" and plan["before_id"] == "b"

    # spine tail: zero-width after the last segment, no right flank
    plan = plan_chunk_insert(gapped, 1)
    assert plan == {"after_id": "b", "before_id": None,
                    "start_s": 9.0, "end_s": 9.0, "welded": True, "rank": 0.0}

    # anchors resolve past synthetics: inhale · um · inhale stack in ONE gap
    # (C.1 drive find — the after/before anchors must be LAYER-0 ids)
    stacked = [seg("a", 0.0, 4.5), seg("ins1", 4.5, 5.0), seg("b", 5.0, 9.0)]
    plan = plan_chunk_insert(stacked, 1, inserted_ids={"ins1"})
    assert plan == {"after_id": "a", "before_id": "b",
                    "start_s": 5.0, "end_s": 5.0, "welded": True, "rank": 0.0}
    # the seam between the real cursor and a synthetic right neighbor works too
    # — and the rank orders the new insert BEFORE the same-start sibling (the
    # split-then-isolate case, FINDING 131ba57a: creation order never could)
    plan = plan_chunk_insert(stacked, 0, inserted_ids={"ins1"})
    assert plan["after_id"] == "a" and plan["before_id"] == "b"
    assert plan["welded"] and plan["start_s"] == 4.5
    assert plan["rank"] == -1.0
    # after a zero-width same-start left sibling: rank goes ABOVE it (a stack
    # built left-to-right keeps arrival order); between two = the midpoint
    zw = [seg("a", 0.0, 4.5), seg("z1", 4.5, 4.5), seg("ins1", 4.5, 5.0),
          seg("b", 5.0, 9.0)]
    plan = plan_chunk_insert(zw, 1, inserted_ids={"z1", "ins1"},
                             insert_ranks={"z1": -1.0, "ins1": 0.0})
    assert plan["rank"] == -0.5
    plan = plan_chunk_insert(zw, 1, inserted_ids={"z1"}, insert_ranks={"z1": -1.0})
    assert plan["rank"] == 0.0
    # no layer-0 segment left of the seam: nothing to anchor
    assert plan_chunk_insert([seg("ins1", 0.0, 1.0)], 0,
                             inserted_ids={"ins1"}) is None

    # overlap beyond the weld eps: refuse (nudge the overlap first)
    overlap = [seg("a", 0.0, 5.2), seg("b", 5.0, 9.0)]
    assert plan_chunk_insert(overlap, 0) is None
    # missing times / out of range
    assert plan_chunk_insert([seg("a", None, None), seg("b", 5.0, 9.0)], 0) is None
    assert plan_chunk_insert([seg("a", 0.0, 4.5), seg("b", None, None)], 0) is None
    assert plan_chunk_insert(gapped, 5) is None


def test_spineview_insert_echo_bookkeeping():
    """Local echo of chunk insertion/removal (hermetic, no graph stack): the
    synthetic segment splices after its flank with the flank's layer-0 index,
    inserted_ids/insert_labels track it, and removal vacates the position."""
    from cjm_transcript_correction_core.models import SpineSegment

    view = SpineView.__new__(SpineView)
    view.segments = [SpineSegment(id="a", index=0, text="one",
                                  start_time=0.0, end_time=4.5),
                     SpineSegment(id="b", index=1, text="two",
                                  start_time=6.0, end_time=9.0)]
    view.inserted_ids = set()
    view.insert_labels = {}
    view.insert_ranks = {}
    view.turn_proposals = {}

    pos = view.add_insert_local(
        {"id": "ins1", "payload": {"operation": "chunk_insert",
                                   "after_segment_id": "a", "start_time": 4.5,
                                   "end_time": 6.0, "label": "inhale", "text": ""}})
    assert pos == 1
    assert [s.id for s in view.segments] == ["a", "ins1", "b"]
    assert view.segments[1].index == 0 and view.segments[1].text == ""
    assert view.inserted_ids == {"ins1"}
    assert view.insert_labels["ins1"] == "inhale"
    assert view.seen_insert_labels == ["inhale"]   # the I-menu's observed tier

    # a SIBLING insert in the same gap lands after the earlier one (start_time
    # order under the shared layer-0 anchor — the stacked inhale·um·inhale echo)
    pos = view.add_insert_local(
        {"id": "ins3", "payload": {"operation": "chunk_insert",
                                   "after_segment_id": "a", "start_time": 6.0,
                                   "end_time": 6.0, "label": "um", "text": ""}})
    assert pos == 2
    assert [s.id for s in view.segments] == ["a", "ins1", "ins3", "b"]
    assert view.remove_insert_local("ins3") == 2

    # a foreign flank refuses the echo (the reload will place it, or drop it)
    assert view.add_insert_local(
        {"id": "ins2", "payload": {"operation": "chunk_insert",
                                   "after_segment_id": "zz"}}) is None

    assert view.remove_insert_local("ins1") == 1
    assert [s.id for s in view.segments] == ["a", "b"]
    assert view.inserted_ids == set() and view.insert_labels == {}
    assert view.seen_insert_labels == []   # a label leaves with its last insert
    assert view.remove_insert_local("ins1") is None


def test_lane_gate_scopes_the_vocabulary():
    """DEC cc55a7b5/8a4df244: one check_action gate — assign lane exposes only
    its vocabulary, assign-only actions are inert in the walk lane, picker
    stages stay gated to walk/open/quit, and the retired reviewed gestures
    (DEC c1bb202f) are gone from the binding roster."""
    from cjm_transcript_correction_tui.app import CorrectionApp
    app = CorrectionApp()
    app.stage = "correct"
    app.lane = "walk"
    assert app.check_action("edit", ()) and app.check_action("next", ())
    assert app.check_action("cycle_lane", ())
    assert not app.check_action("assign_same", ())
    assert not app.check_action("assign_pick", ())
    app.lane = "assign"
    assert app.check_action("assign_pick", ()) and app.check_action("assign_new", ())
    assert app.check_action("next", ()) and app.check_action("cycle_lane", ())
    assert not app.check_action("edit", ()) and not app.check_action("insert_chunk", ())
    assert not app.check_action("mark_quick", ())
    app.stage = "select"
    assert app.check_action("next", ()) and not app.check_action("assign_pick", ())
    actions = {b.action for b in app.BINDINGS}
    assert "reviewed" not in actions and "unreview" not in actions
    assert "assign_same" in {a.split("(")[0] for a in actions}


def test_parse_entity_input_provisional():
    """DEC 484e2d74: a leading ? marks a descriptive PROVISIONAL handle."""
    from cjm_transcript_correction_tui.spine import parse_entity_input
    assert parse_entity_input("Dan Carlin") == ("Dan Carlin", False)
    assert parse_entity_input("? HH montage narrator") == ("HH montage narrator", True)
    assert parse_entity_input("?HH promo voice A") == ("HH promo voice A", True)
    assert parse_entity_input("?") is None
    assert parse_entity_input("   ") is None


def test_assign_menu_layers_source_speakers_first():
    """DEC 4ec6a49c: digit menu = this source's assigned speakers (encounter
    order) then the rest of the registry; provisional names read with ?."""
    from cjm_transcript_correction_core.models import SpineSegment
    from cjm_transcript_correction_tui.app import CorrectionApp
    from cjm_transcript_correction_tui.spine import SpineView
    app = CorrectionApp()
    app._entities = [
        {"id": "e-b", "properties": {"canonical_name": "Bob"}},
        {"id": "e-a", "properties": {"canonical_name": "Alice"}},
        {"id": "e-n", "properties": {"canonical_name": "HH montage narrator",
                                     "provisional": True}}]
    view = SpineView.__new__(SpineView)
    view.segments = [SpineSegment(id=f"s{i}", index=i, text="t") for i in range(3)]
    view.speakers = {"s2": {"entity_id": "e-n", "verdict": "name", "correction_id": "c1"},
                     "s0": {"entity_id": "e-a", "verdict": "name", "correction_id": "c2"}}
    app.view = view
    menu = app._assign_menu()
    assert [m[0] for m in menu] == ["e-a", "e-n", "e-b"]
    assert menu[1][1] == "?HH montage narrator"
    assert app._entity_name("e-a") == "Alice"


def test_plan_chunk_split_caret_seed_and_anchors():
    """The S gesture unit (work item 99c1d2ba): caret partitions the text
    (whitespace-normalized halves), the time seed interpolates the caret
    fraction and clamps strictly inside the span, anchors resolve past
    synthetics to layer-0 flanks, and edge-of-text carets refuse (that is a
    nudge, not a split)."""
    from types import SimpleNamespace

    from cjm_transcript_correction_tui.spine import plan_chunk_split

    segs = [SimpleNamespace(id="a", text="alpha beta gamma",
                            start_time=10.0, end_time=14.0),
            SimpleNamespace(id="b", text="tail", start_time=14.0, end_time=16.0)]
    # caret after "alpha " (6 of 16 chars): halves strip the boundary space,
    # seed = 10 + 4 * 6/16 = 11.5, boundary words bank the flywheel context
    plan = plan_chunk_split(segs, 0, 6)
    assert plan is not None
    assert (plan["left_text"], plan["right_text"]) == ("alpha", "beta gamma")
    assert abs(plan["split_s"] - 11.5) < 1e-9 and plan["end_s"] == 14.0
    assert (plan["after_id"], plan["before_id"]) == ("a", "b")
    assert plan["boundary_words"] == {"left": "alpha", "right": "beta"}

    # the submitted text overrides the segment text (a typo fixed in the editor)
    fixed = plan_chunk_split(segs, 0, 6, text="alfax beta gamma")
    assert fixed["left_text"] == "alfax"

    # edge-of-text carets refuse: both halves must keep words
    assert plan_chunk_split(segs, 0, 0) is None
    assert plan_chunk_split(segs, 0, len(segs[0].text)) is None
    assert plan_chunk_split(segs, 0, 3) is not None   # mid-word caret still splits
    # missing times / off-spine refuse
    assert plan_chunk_split(segs, 5, 3) is None
    assert plan_chunk_split([SimpleNamespace(id="x", text="a b", start_time=None,
                                             end_time=None)], 0, 1) is None

    # splitting a SYNTHETIC: anchors resolve past it to the layer-0 flanks
    synth = SimpleNamespace(id="ins-1", text="um inhale", start_time=14.2,
                            end_time=15.0)
    walked = [segs[0], synth, segs[1]]
    p2 = plan_chunk_split(walked, 1, 2, inserted_ids={"ins-1"})
    assert p2 is not None
    assert p2["segment_id"] == "ins-1"
    assert (p2["after_id"], p2["before_id"]) == ("a", "b")

    # the clamp keeps the seed strictly inside a short span (no collapsed half)
    tiny = [SimpleNamespace(id="t", text="hm um", start_time=0.0, end_time=0.02)]
    pt = plan_chunk_split(tiny, 0, 2)
    assert pt is not None and 0.0 < pt["split_s"] < 0.02


def test_spineview_split_echo():
    """Local echo of a chunk split (hermetic): the target keeps the LEFT half
    (text truncated, end pulled to the cut) and the right half splices as a
    synthetic sibling directly after it — welded at split_s, exactly what a
    projection reload composes."""
    from cjm_transcript_correction_core.models import SpineSegment

    view = SpineView.__new__(SpineView)
    view.segments = [SpineSegment(id="a", index=0, text="alpha beta gamma",
                                  start_time=0.0, end_time=6.0),
                     SpineSegment(id="b", index=1, text="tail",
                                  start_time=6.0, end_time=9.0)]
    view.inserted_ids = set()
    view.insert_labels = {}
    view.insert_ranks = {}
    view.turn_proposals = {}
    view._turns = []

    pos = view.split_local(0, "alpha", 2.0,
                           {"id": "sp1", "payload": {"operation": "chunk_insert",
                                                     "after_segment_id": "a",
                                                     "start_time": 2.0, "end_time": 6.0,
                                                     "label": None,
                                                     "text": "beta gamma"}})
    assert pos == 1
    assert [s.id for s in view.segments] == ["a", "sp1", "b"]
    assert (view.segments[0].text, view.segments[0].end_time) == ("alpha", 2.0)
    assert (view.segments[1].text, view.segments[1].start_time,
            view.segments[1].end_time) == ("beta gamma", 2.0, 6.0)
    assert "sp1" in view.inserted_ids

    # splitting the SYNTHETIC right half echoes uniformly: the new piece walks
    # past its sibling under the shared layer-0 anchor (start_time order)
    pos = view.split_local(1, "beta", 4.0,
                           {"id": "sp2", "payload": {"operation": "chunk_insert",
                                                     "after_segment_id": "a",
                                                     "start_time": 4.0, "end_time": 6.0,
                                                     "label": None, "text": "gamma"}})
    assert pos == 2
    assert [s.id for s in view.segments] == ["a", "sp1", "sp2", "b"]
    assert [(s.text, s.start_time, s.end_time) for s in view.segments] == [
        ("alpha", 0.0, 2.0), ("beta", 2.0, 4.0), ("gamma", 4.0, 6.0),
        ("tail", 6.0, 9.0)]


def test_spineview_rank_echo_and_unsplit():
    """131ba57a echoes (hermetic): (a) a rank -1 insert lands BETWEEN a split's
    halves even though the right half echoed first; (b) unsplit_local removes
    the right half AND restores the target's pre-split text/end from the
    registered group snapshot."""
    from cjm_transcript_correction_core.models import SpineSegment

    view = SpineView.__new__(SpineView)
    view.segments = [SpineSegment(id="a", index=0, text="alpha beta gamma",
                                  start_time=0.0, end_time=6.0),
                     SpineSegment(id="b", index=1, text="tail",
                                  start_time=6.0, end_time=9.0)]
    view.inserted_ids = set()
    view.insert_labels = {}
    view.insert_ranks = {}
    view.split_groups = {}
    view.turn_proposals = {}
    view._turns = []

    # split echo (right half at the weld), then the user's inhale insert from
    # the LEFT half: rank -1 overtakes the same-start sibling
    assert view.split_local(0, "alpha", 2.0,
                            {"id": "sp1", "payload": {"operation": "chunk_insert",
                                                      "after_segment_id": "a",
                                                      "start_time": 2.0,
                                                      "end_time": 6.0,
                                                      "text": "beta gamma"}}) == 1
    view.split_groups["sp1"] = {"group_ids": ["t1", "n1"], "target_id": "a",
                                "old_text": "alpha beta gamma", "old_end": 6.0}
    pos = view.add_insert_local(
        {"id": "inh", "payload": {"operation": "chunk_insert",
                                  "after_segment_id": "a", "start_time": 2.0,
                                  "end_time": 2.0, "label": "inhale",
                                  "text": "", "rank": -1.0}})
    assert pos == 1
    assert [s.id for s in view.segments] == ["a", "inh", "sp1", "b"]
    # rank 0 (absent) still lands after the same-start sibling stack
    pos = view.add_insert_local(
        {"id": "um", "payload": {"operation": "chunk_insert",
                                 "after_segment_id": "a", "start_time": 2.0,
                                 "end_time": 2.0, "text": ""}})
    assert pos == 3
    assert [s.id for s in view.segments] == ["a", "inh", "sp1", "um", "b"]
    assert view.remove_insert_local("um") == 3
    assert view.remove_insert_local("inh") == 1

    # unsplit: right half leaves, the target gets its pre-split text/end back
    assert view.unsplit_local("sp1") == 1
    assert [s.id for s in view.segments] == ["a", "b"]
    assert (view.segments[0].text, view.segments[0].end_time) == \
        ("alpha beta gamma", 6.0)
    assert view.split_groups == {} and view.insert_ranks == {}


def test_aseg_index_synthetics_inherit_their_anchor_side():
    """Drive find 2026-07-27: coarse seams often cut flush with the last
    chunk's end, so a bookend insert born AT the seam time would time-bisect
    into the NEXT audio segment and paint under its banner. Synthetic chunks
    inherit the aseg of the layer-0 segment they follow; layer-0 attribution
    stays a pure time bisect."""
    from cjm_transcript_correction_core.models import SpineSegment

    view = SpineView.__new__(SpineView)
    view._aseg_starts = [0.0, 100.0]
    view.segments = [
        SpineSegment(id="a", index=0, text="last of aseg 1",
                     start_time=90.0, end_time=100.0),
        SpineSegment(id="inh", index=0, text="",       # bookend insert AT the seam
                     start_time=100.0, end_time=100.0),
        SpineSegment(id="b", index=1, text="first of aseg 2",
                     start_time=100.4, end_time=105.0),
    ]
    view.inserted_ids = {"inh"}
    assert view.aseg_index(0) == 0
    assert view.aseg_index(1) == 0    # inherited from "a", NOT bisected into aseg 2
    assert view.aseg_index(2) == 1    # the banner moves to the layer-0 opener
    # a synthetic with no layer-0 left neighbor falls back to the time bisect
    view.segments.insert(0, SpineSegment(id="head", index=0, text="",
                                         start_time=0.0, end_time=0.0))
    view.inserted_ids.add("head")
    assert view.aseg_index(0) == 0


def test_split_halves_get_turn_proposals_mid_session():
    """Drive find 2026-07-27 (assign lane at ~760s): proposals were a LOAD-time
    map, so mid-session split halves painted ∅ until a restart. The echoes now
    refresh id-scoped: text-bearing halves propose immediately, empty inserts
    never do (text = the unit of attribution supervision), and removal drops
    the stale chip."""
    from cjm_transcript_correction_core.models import SpineSegment
    from cjm_transcript_correction_core.signals import speaker_turn_proposals

    view = SpineView.__new__(SpineView)
    view.segments = [SpineSegment(id="a", index=0, text="alpha beta gamma",
                                  start_time=0.0, end_time=6.0),
                     SpineSegment(id="b", index=1, text="tail",
                                  start_time=6.0, end_time=9.0)]
    view.inserted_ids = set()
    view.insert_labels = {}
    view.insert_ranks = {}
    view.split_groups = {}
    view._turns = [{"start": 0.0, "end": 3.5, "speaker": "SPEAKER_00"},
                   {"start": 3.5, "end": 9.0, "speaker": "SPEAKER_01"}]
    view.turn_proposals = speaker_turn_proposals(view.segments, view._turns)
    assert view.turn_proposals["a"]["cluster"] == "SPEAKER_00"

    # split at 2.0: both halves re-propose — the left keeps S00, the right
    # (2.0-6.0, text-bearing) proposes S01 immediately, no restart needed
    view.split_local(0, "alpha", 2.0,
                     {"id": "sp1", "payload": {"operation": "chunk_insert",
                                               "after_segment_id": "a",
                                               "start_time": 2.0, "end_time": 6.0,
                                               "text": "beta gamma"}})
    assert view.turn_proposals["a"]["cluster"] == "SPEAKER_00"
    assert view.turn_proposals["sp1"]["cluster"] == "SPEAKER_01"

    # an EMPTY insert at the weld never proposes, even under full coverage
    view.add_insert_local({"id": "inh", "payload": {"operation": "chunk_insert",
                                                    "after_segment_id": "a",
                                                    "start_time": 2.0, "end_time": 2.3,
                                                    "label": "inhale", "text": "",
                                                    "rank": -1.0}})
    view.refresh_turn_proposal("inh")
    assert "inh" not in view.turn_proposals
    # text landing later (the e-edit lane) flips eligibility on refresh
    view.segments[1].text = "recovered words"
    view.refresh_turn_proposal("inh")
    assert view.turn_proposals["inh"]["cluster"] == "SPEAKER_00"
    # removal drops the chip with the row
    assert view.remove_insert_local("inh") == 1
    assert "inh" not in view.turn_proposals


def test_plan_gate_grammar():
    """The F gesture unit (DEC 8e05b87b): w = watermark at the pause point,
    signoff = end-of-source, exclude/resume keep the watermark, junk refuses."""
    from cjm_transcript_correction_tui.spine import plan_gate

    assert plan_gate("w", 2016.2, 2500.0, None) == ("in_progress", 2016.2)
    assert plan_gate("w 100.5", None, 2500.0, None) == ("in_progress", 100.5)  # explicit override needs no cursor time
    assert plan_gate("signoff", 2016.2, 2500.0, 2016.2) == ("signed_off", 2500.0)
    assert plan_gate("exclude", 2016.2, 2500.0, 2016.2) == ("excluded", 2016.2)
    assert plan_gate("resume", None, None, 2016.2) == ("in_progress", 2016.2)
    assert plan_gate("", 1.0, 2.0, None) is None            # empty
    assert plan_gate("done", 1.0, 2.0, None) is None        # unknown verb
    assert plan_gate("w abc", 1.0, 2.0, None) is None       # non-numeric override
    assert plan_gate("w", None, 2.0, None) is None          # no time to anchor
    assert plan_gate("signoff", 1.0, None, None) is None    # no source end


def test_segment_word_tokens_offsets():
    """The annotate lane's selection unit: whitespace words WITH char offsets."""
    from cjm_transcript_correction_tui.spine import segment_word_tokens
    assert segment_word_tokens("um I  mean") == [(0, 2, "um"), (3, 4, "I"), (6, 10, "mean")]
    assert segment_word_tokens("") == [] and segment_word_tokens("   ") == []


def test_snap_word_span_direct_fuzzy_and_estimated():
    """fc42614d: the span DERIVES from FA word times — equal counts map by
    position, edited text aligns by normalized tokens, no cache estimates."""
    from cjm_transcript_correction_tui.spine import segment_word_tokens, snap_word_span
    toks = segment_word_tokens("um I mean")
    fa = [{"s": 10.0, "e": 10.3, "text": "um"},
          {"s": 10.35, "e": 10.5, "text": "i"},
          {"s": 10.6, "e": 11.0, "text": "mean"}]
    # direct positional map (counts equal; punctuation/case-insensitive)
    s, e, snap, hit = snap_word_span(toks, 0, 1, 10.0, 11.0, len("um I mean"), fa)
    assert (s, e, snap) == (10.0, 10.5, "fa-word") and len(hit) == 2
    # fuzzy: the effective text carries an extra edited word FA never saw —
    # the unmatched ENDPOINT downgrades the stamp (finding 162935f5): the
    # matched edge holds, the missing edge extrapolates at local rate
    toks2 = segment_word_tokens("um I really mean")
    s, e, snap, hit = snap_word_span(toks2, 2, 3, 10.0, 11.0, len("um I really mean"), fa)
    assert snap == "fa-partial" and e == 11.0 and 10.0 <= s < 10.6
    # a fully-matched sub-range still earns fa-word (interior gaps only)
    s, e, snap, hit = snap_word_span(toks2, 3, 3, 10.0, 11.0, len("um I really mean"), fa)
    assert snap == "fa-word" and (s, e) == (10.6, 11.0)
    # no cache -> char-fraction estimation (the split-seed regime)
    s, e, snap, hit = snap_word_span(toks, 0, 0, 10.0, 11.0, len("um I mean"), None)
    assert snap == "estimated" and hit == [] and 10.0 <= s < e <= 11.0
    # words outside the segment window are not candidates (window filter)
    far = [{"s": 50.0, "e": 50.5, "text": "um"}]
    assert snap_word_span(toks, 0, 0, 10.0, 11.0, 9, far)[2] == "estimated"
    # refusals: no tokens / bad range / collapsed span
    assert snap_word_span([], 0, 0, 10.0, 11.0, 9, fa) is None
    assert snap_word_span(toks, 2, 1, 10.0, 11.0, 9, fa) is None
    assert snap_word_span(toks, 0, 0, 11.0, 11.0, 9, fa) is None


def test_snap_word_span_endpoint_partial_live_case():
    """Finding 162935f5 regression, on the live CW #57 shape: fast 'you know'
    where FA dropped 'know' — the old code committed the narrowed one-word
    span stamped fa-word; now the tail extrapolates and the stamp says
    fa-partial (order can never invert: extrapolation only grows the span)."""
    from cjm_transcript_correction_tui.spine import segment_word_tokens, snap_word_span
    toks = segment_word_tokens("you know, you could debate")
    fa = [{"s": 185.92, "e": 186.08, "text": "you"},
          {"s": 186.25, "e": 186.32, "text": "you"},
          {"s": 186.32, "e": 186.50, "text": "could"},
          {"s": 186.50, "e": 186.80, "text": "debate"}]
    s, e, snap, hit = snap_word_span(toks, 0, 1, 185.9, 186.8, len("you know, you could debate"), fa)
    assert snap == "fa-partial" and [w["text"] for w in hit] == ["you"]
    assert s == 185.92          # matched head edge holds
    assert e > 186.08           # tail GREW past the smeared FA edge (no narrowing)
    assert e <= 186.8           # clamped to the segment window
    # leading-edge dual: selection whose HEAD is the unmatched token
    toks2 = segment_word_tokens("know, you could")
    fa2 = [{"s": 186.25, "e": 186.32, "text": "you"},
          {"s": 186.32, "e": 186.50, "text": "could"}]
    s, e, snap, hit = snap_word_span(toks2, 0, 2, 186.1, 186.6, len("know, you could"), fa2)
    assert snap == "fa-partial" and e == 186.50
    assert 186.1 <= s < 186.25  # head grew backward, floored at the segment start


def test_annotate_lane_gate_and_selection_range():
    """The annotate lane scopes its vocabulary through the one check_action
    gate (fc42614d): annotate-only actions are inert in the walk lane, the
    walk verbs that would mutate the spine are inert in the annotate lane."""
    from cjm_transcript_correction_tui.app import CorrectionApp
    app = CorrectionApp()
    app.stage = "correct"
    app.lane = "annotate"
    for a in ("word_left", "word_select", "annotate_pick", "annotate_quick",
              "overlay_remove", "overlay_nudge", "nudge_step_up", "next_overlay",
              "next", "replay", "cycle_lane", "cycle_lane_prev"):
        assert app.check_action(a, ()), a
    for a in ("edit", "insert_chunk", "mark_quick", "shift_push", "split_chunk",
              "remove_insert", "assign_pick", "propose_accept",
              "nudge_end_earlier"):
        assert not app.check_action(a, ()), a
    app.lane = "walk"
    for a in ("word_left", "annotate_pick", "annotate_quick", "overlay_remove",
              "overlay_nudge"):
        assert not app.check_action(a, ()), a
    assert app.check_action("edit", ()) and app.check_action("remove_insert", ())
    # shift+tab (reverse cycle) is live in every lane; the boundary time-nudge
    # verbs stay walk/propose vocabulary — overlay spans refine by supersede
    for lane in ("walk", "assign", "propose"):
        app.lane = lane
        assert app.check_action("cycle_lane_prev", ()), lane
    app.lane = "walk"
    # selection: cursor-word when unanchored, inclusive clamped range when anchored
    app._word_cursor, app._word_anchor = 2, None
    assert app._selection_range(3) == (2, 2)
    app._word_anchor = 0
    assert app._selection_range(3) == (0, 2)
    app._word_cursor = 99          # clamps
    assert app._selection_range(3) == (0, 2)
    assert app._selection_range(0) is None


def test_spineview_overlay_bookkeeping():
    """SpineView overlay state: echoes recompute the ◈ set, per-segment lookup
    is anchor-scoped, labels/count derive from the ACTIVE overlays."""
    from cjm_transcript_correction_tui.spine import SpineView
    view = SpineView.__new__(SpineView)
    view._overlays = []
    view._recompute_overlay_ids()
    assert view.overlay_ids == set() and view.overlay_count == 0
    o1 = {"id": "o1", "correction_type": "annotation",
          "payload": {"operation": "speech_overlay", "label": "hesitation-marker",
                      "anchor": {"kind": "span", "segment_id": "a",
                                 "char_start": 0, "char_end": 2, "text_snapshot": "um"}}}
    o2 = {"id": "o2", "correction_type": "annotation",
          "payload": {"operation": "speech_overlay", "label": "word-repeat",
                      "anchor": {"kind": "span", "segment_id": "b",
                                 "char_start": 3, "char_end": 7, "text_snapshot": "very"}}}
    view.add_overlay_local(o1)
    view.add_overlay_local(o2)
    assert view.overlay_ids == {"a", "b"} and view.overlay_count == 2
    assert [o["id"] for o in view.overlays_for("a")] == ["o1"]
    assert view.seen_overlay_labels == ["hesitation-marker", "word-repeat"]
    view.remove_overlay_local("o1")
    assert view.overlay_ids == {"b"} and view.overlays_for("a") == []


def test_overlay_at_cursor_targeting():
    """Gesture targeting: covering overlay wins; the newest-fallback serves
    nudge/remove but is DROPPED for audition (covering_only) so R on an
    uncovered word stays a selection preview."""
    from cjm_transcript_correction_core.models import SpineSegment
    from cjm_transcript_correction_tui.app import CorrectionApp
    from cjm_transcript_correction_tui.spine import SpineView
    app = CorrectionApp()
    view = SpineView.__new__(SpineView)
    o_um = {"id": "o1", "correction_type": "annotation",
            "payload": {"operation": "speech_overlay", "label": "hesitation-marker",
                        "anchor": {"kind": "span", "segment_id": "a",
                                   "char_start": 0, "char_end": 2,
                                   "text_snapshot": "um"}}}
    o_mean = {"id": "o2", "correction_type": "annotation",
              "payload": {"operation": "speech_overlay", "label": "word-repeat",
                          "anchor": {"kind": "span", "segment_id": "a",
                                     "char_start": 5, "char_end": 9,
                                     "text_snapshot": "mean"}}}
    view._overlays = [o_um, o_mean]
    view._recompute_overlay_ids()
    app.view = view
    seg = SpineSegment(id="a", index=0, text="um I mean")
    app._word_cursor = 0                       # on "um" — o1 covers it
    assert app._overlay_at_cursor(seg)["id"] == "o1"
    assert app._overlay_at_cursor(seg, covering_only=True)["id"] == "o1"
    app._word_cursor = 1                       # on "I" — nothing covers it
    assert app._overlay_at_cursor(seg)["id"] == "o2"          # newest-fallback (nudge/remove)
    assert app._overlay_at_cursor(seg, covering_only=True) is None  # audition previews instead


def test_overlay_pick_overrides_covering_test():
    """The o-cycled ◈ pick (finding 13c5f6fb): an explicit id-level target wins
    over the covering test in BOTH modes — a drifted-anchor or shadowed
    overlay stays reachable for R/x/nudges — and a stale pick (superseded or
    removed id) clears itself and falls back to covering."""
    from cjm_transcript_correction_core.models import SpineSegment
    from cjm_transcript_correction_tui.app import CorrectionApp
    from cjm_transcript_correction_tui.spine import SpineView
    app = CorrectionApp()
    view = SpineView.__new__(SpineView)
    # o1's anchor drifted onto a word FRAGMENT (covers no full token); o2, the
    # newer stacked overlay, covers "um" — the covering test can never pick o1.
    o_drift = {"id": "o1", "correction_type": "annotation",
               "payload": {"operation": "speech_overlay", "label": "hesitation-marker",
                           "start_time": 1.0, "end_time": 1.2,
                           "anchor": {"kind": "span", "segment_id": "a",
                                      "char_start": 3, "char_end": 4,
                                      "text_snapshot": "I"}}}
    o_cover = {"id": "o2", "correction_type": "annotation",
               "payload": {"operation": "speech_overlay", "label": "word-repeat",
                           "start_time": 0.0, "end_time": 0.4,
                           "anchor": {"kind": "span", "segment_id": "a",
                                      "char_start": 0, "char_end": 2,
                                      "text_snapshot": "um"}}}
    view._overlays = [o_drift, o_cover]
    view._recompute_overlay_ids()
    app.view = view
    seg = SpineSegment(id="a", index=0, text="um I mean")
    app._word_cursor = 0                       # on "um" — o2 covers, o1 unreachable
    assert app._overlay_at_cursor(seg, covering_only=True)["id"] == "o2"
    app._overlay_pick = "o1"                   # the explicit pick reaches it anyway
    assert app._overlay_at_cursor(seg)["id"] == "o1"
    assert app._overlay_at_cursor(seg, covering_only=True)["id"] == "o1"
    app._overlay_pick = "gone"                 # superseded/removed pick self-clears
    assert app._overlay_at_cursor(seg)["id"] == "o2"
    assert app._overlay_pick is None
    # the cycle enumerates in TIME order (o1 starts later despite listing first)
    assert [o["id"] for o in app._segment_overlays_by_time(seg)] == ["o2", "o1"]
    # the cycle verb rides the annotate lane gate
    app.stage, app.lane = "correct", "annotate"
    assert app.check_action("overlay_cycle", ())
    app.lane = "walk"
    assert not app.check_action("overlay_cycle", ())


def test_neighbor_word_bound_overshoot_guard_data():
    """The overlay-nudge advisory's data (pure): the facing FA boundary of the
    span's in-segment neighbor; None off the segment edge or when the
    neighbor's time is not FA-anchored (estimation is no guard)."""
    from cjm_transcript_correction_tui.spine import neighbor_word_bound, segment_word_tokens
    text = "uh in your"
    toks = segment_word_tokens(text)
    # FA aligned the pre-edit text — "uh" has no row (the restored-word case)
    fa = [{"s": 32.60, "e": 32.75, "text": "in"},
          {"s": 32.75, "e": 33.10, "text": "your"}]
    span = (0, 2)  # the "uh" anchor's char range
    # growing the end: the guard boundary is "in"'s FA start
    assert neighbor_word_bound(toks, *span, "next", 32.3, 33.5, len(text), fa) == ("in", 32.60)
    # shrinking toward the head: no in-segment word before "uh"
    assert neighbor_word_bound(toks, *span, "prev", 32.3, 33.5, len(text), fa) is None
    # a span on "your": prev neighbor "in" faces with its FA END
    span_your = (toks[2][0], toks[2][1])
    assert neighbor_word_bound(toks, *span_your, "prev", 32.3, 33.5, len(text), fa) == ("in", 32.75)
    assert neighbor_word_bound(toks, *span_your, "next", 32.3, 33.5, len(text), fa) is None
    # no FA cache -> every neighbor estimates -> no guard data
    assert neighbor_word_bound(toks, *span, "next", 32.3, 33.5, len(text), None) is None
