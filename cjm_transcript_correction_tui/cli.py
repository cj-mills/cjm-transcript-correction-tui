import argparse

from .app import CorrectionApp


def build_parser() -> argparse.ArgumentParser:  # Configured CLI parser
    """The TUI driver's argument surface (mirrors correction-core's run/review args)."""
    p = argparse.ArgumentParser(
        prog="cjm-transcript-correction-tui",
        description="Keyboard-first correction loop over a transcription context graph "
                    "(document-order segment walk, VAD-chunk auto-play, fidelity edits).")
    p.add_argument("--graph-db-path", required=True,
                   help="The shared transcription graph db (the committed spine)")
    p.add_argument("--source", default=None,
                   help="Source node id or title substring (required when the graph "
                        "holds more than one Source)")
    p.add_argument("--manifests-dir", default=".cjm/manifests",
                   help="Capability manifests directory")
    p.add_argument("--rendition", default=None,
                   help="AudioRendition selector when a source has more than one "
                        "(\"raw\" or a preprocessing substring); default: auto-select")
    p.add_argument("--actor", default="human",
                   help="Actor recorded on corrections + review markers")
    p.add_argument("--no-autoplay", action="store_true",
                   help="Do not auto-play the focused segment's VAD chunk")
    return p


def main() -> int:  # Console-script entry point
    """Parse args, run the correction loop (the app owns the event loop + teardown)."""
    args = build_parser().parse_args()
    app = CorrectionApp(args.graph_db_path, source=args.source,
                        manifests_dir=args.manifests_dir, rendition=args.rendition,
                        actor=args.actor, autoplay=not args.no_autoplay)
    app.run()
    return 0
