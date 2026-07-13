"""Tests for the [ / ] boundary-shift gesture planning + unprune bookkeeping.

Hermetic: plan_boundary_shift is pure; the SpineView prune bookkeeping is
exercised on a directly-constructed view (no graph stack)."""
from cjm_transcript_correction_tui.spine import SpineView, plan_boundary_shift


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
