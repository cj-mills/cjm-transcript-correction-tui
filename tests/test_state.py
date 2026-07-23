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


def test_state_spine_choice_merges_with_cursor(tmp_path):
    # The spine choice (DEC f1024568) and the cursor bookmark share one
    # per-source entry — neither write may clobber the other.
    db = str(tmp_path / "g.db")
    save_tui_state(db, "src-1", 42)
    save_tui_state(db, "src-1", None, skeleton="sha256:abc", spines=2)
    entry = load_tui_state(db)["src-1"]
    assert entry["cursor"] == 42
    assert entry["skeleton"] == "sha256:abc" and entry["spines"] == 2
    # A later cursor write keeps the spine choice.
    save_tui_state(db, "src-1", 99)
    entry = load_tui_state(db)["src-1"]
    assert entry["cursor"] == 99 and entry["skeleton"] == "sha256:abc"


def test_spine_picker_labels_and_selectors():
    from cjm_transcript_correction_tui.app import selector_for_spine, spine_label
    legacy = {"skeleton_hash": None, "split_policy": None, "segments": 950}
    split = {"skeleton_hash": "sha256:abc123def456", "split_policy": "sentence-split/v1",
             "segments": 1100}
    assert spine_label(legacy) == "vad-only (pre-split)"
    assert spine_label(split) == "sentence-split/v1 · abc123de"
    # The persisted selector round-trips through correction-core's resolver
    # vocabulary: full hash, or the legacy token.
    assert selector_for_spine(split) == "sha256:abc123def456"
    assert selector_for_spine(legacy) == "legacy"
