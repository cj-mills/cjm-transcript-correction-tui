import argparse
import os

from cjm_substrate.core.workspace import resolve_workspace

from .app import CorrectionApp


def build_parser() -> argparse.ArgumentParser:  # Configured CLI parser
    """The TUI driver's argument surface (mirrors correction-core's run/review args)."""
    p = argparse.ArgumentParser(
        prog="cjm-transcript-correction-tui",
        description="Keyboard-first correction loop over a transcription context graph "
                    "(document-order segment walk, VAD-chunk auto-play, fidelity edits).")
    p.add_argument("--graph-db-path", default=None,
                   help="The shared transcription graph db (the committed spine); "
                        "default: the graph capability's persisted config — under an "
                        "active workspace the config store is workspace-scoped, so "
                        "the workspace names the db (2ce81638)")
    p.add_argument("--source", default=None,
                   help="Source node id or title substring; omitted or ambiguous -> "
                        "the in-TUI source picker (correction status at a glance)")
    p.add_argument("--manifests-dir", default=None,
                   help="Capability manifests directory (default: the workspace's "
                        ".cjm/manifests when one is active, else .cjm/manifests under the cwd)")
    p.add_argument("--workspace", default=None,
                   help="Workspace root (5daadfc4; default: CJM_WORKSPACE env, else upward walk "
                        "from cwd). Supplies the manifests default and is exported so capability "
                        "workers resolve workspace-scoped paths and the config store "
                        "supplies the graph db default (2ce81638 discovery is built: "
                        "no --source -> in-TUI picker)")
    p.add_argument("--rendition", default=None,
                   help="AudioRendition selector when a source has more than one "
                        "(\"raw\" or a preprocessing substring); default: auto-select")
    p.add_argument("--skeleton", default=None,
                   help="Skeleton-spine selector when several coexist under one rendition "
                        "(sentence-split, DEC f1024568): \"legacy\" or a skeleton-hash prefix; "
                        "default: the in-TUI spine picker (choice persists in the sidecar)")
    p.add_argument("--actor", default="human",
                   help="Actor recorded on corrections + review markers")
    p.add_argument("--no-autoplay", action="store_true",
                   help="Do not auto-play the focused segment's VAD chunk")
    p.add_argument("--audio-device", default=None,
                   help="Output device index or name substring (default: the system "
                        "default sink — pipewire/pulse routing when available)")
    p.add_argument("--no-resume", action="store_true",
                   help="Start at segment 0 instead of the source's last-focused segment")
    p.add_argument("--shift-floor-ms", type=int, default=0,
                   help="Minimum milliseconds between held-key boundary shifts; 0 = ungoverned "
                        "(the async commit guard is the real governor — a 1ms floor read as "
                        "residual keystroke latency in the 2026-07-14 drive). "
                        "Measure key rates with tests_manual/keyrate_probe.py")
    p.add_argument("--nudge-step-ms", type=float, default=None,
                   help="Boundary time-nudge step per ,/. (end) or </> (start) press, "
                        "milliseconds. Adjustable IN-TUI with { } along the "
                        "5/10/20/50/100/200/500 ladder (the choice persists in the "
                        "sidecar); this flag overrides the persisted preference "
                        "(default: sidecar, else 100)")
    return p


def main() -> int:  # Console-script entry point
    """Parse args, run the correction loop (the app owns the event loop + teardown)."""
    args = build_parser().parse_args()
    # 5daadfc4 workspace: resolve before anything reads paths; export so
    # capability workers (ffmpeg etc.) are workspace-scoped.
    ws = resolve_workspace(explicit=args.workspace)
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")
    device = args.audio_device
    if device is not None and device.isdigit():
        device = int(device)
    app = CorrectionApp(args.graph_db_path, source=args.source,
                        manifests_dir=args.manifests_dir, rendition=args.rendition,
                        skeleton=args.skeleton,
                        actor=args.actor, autoplay=not args.no_autoplay,
                        audio_device=device, resume=not args.no_resume,
                        shift_floor_s=args.shift_floor_ms / 1000.0,
                        nudge_step_ms=args.nudge_step_ms)
    app.run()
    return 0
