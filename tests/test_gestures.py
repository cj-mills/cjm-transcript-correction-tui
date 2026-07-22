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
