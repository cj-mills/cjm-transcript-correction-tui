"""Tests for the last-focused-segment sidecar state (resume across restarts)."""
from cjm_transcript_correction_tui.app import load_tui_state, save_tui_state


def test_state_round_trip(tmp_path):
    db = str(tmp_path / "context_graph.db")
    assert load_tui_state(db) == {}  # absent file -> empty

    save_tui_state(db, "src-1", 233)
    save_tui_state(db, "src-2", 7)
    state = load_tui_state(db)
    assert state["src-1"]["cursor"] == 233 and state["src-2"]["cursor"] == 7
    assert (tmp_path / "context_graph.db.tui-state.json").exists()

    save_tui_state(db, "src-1", 310)  # overwrite merges, does not clobber others
    state = load_tui_state(db)
    assert state["src-1"]["cursor"] == 310 and state["src-2"]["cursor"] == 7


def test_state_corrupt_file_tolerated(tmp_path):
    db = str(tmp_path / "g.db")
    (tmp_path / "g.db.tui-state.json").write_text("{not json")
    assert load_tui_state(db) == {}
    save_tui_state(db, "s", 1)  # recovers by rewriting
    assert load_tui_state(db)["s"]["cursor"] == 1
