"""Measure YOUR key delivery rates to tune --shift-floor-ms (run in a real terminal).

    python tests_manual/keyrate_probe.py

Protocol: HOLD the down arrow ~3 seconds (measures the OS autorepeat interval),
then TAP the right arrow as fast as you can ~10 times (measures your burst tap
rate), then press q for the report. Read it as: the shift floor should sit at
or just above the largest interval you WANT honored — e.g. autorepeat p50 to
accept every held-key repeat (if the screen keeps up), or your tap p50 to
honor deliberate taps while damping autorepeat."""
import time

from textual.app import App, ComposeResult
from textual.widgets import Static


class KeyRateProbe(App):
    """Records inter-key intervals per key; q exits with a per-key report."""

    def __init__(self):
        super().__init__()
        self.events = []  # (key, monotonic time)

    def compose(self) -> ComposeResult:
        yield Static("hold ↓ ~3s · tap → fast ~10x · q = report", id="s")

    def on_key(self, event) -> None:
        t = time.monotonic()
        if event.key == "q":
            self.exit(self._report())
            return
        self.events.append((event.key, t))
        self.query_one("#s", Static).update(
            f"{len(self.events)} events · last: {event.key}")

    def _report(self) -> dict:
        out = {}
        for key in {k for k, _ in self.events}:
            ts = [t for k, t in self.events if k == key]
            gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
            if gaps:
                out[key] = {"n": len(ts),
                            "min_ms": round(gaps[0] * 1000, 1),
                            "p50_ms": round(gaps[len(gaps) // 2] * 1000, 1),
                            "max_ms": round(gaps[-1] * 1000, 1)}
        return out


if __name__ == "__main__":
    report = KeyRateProbe().run()
    print("\nper-key intervals (ms):")
    for key, stats in (report or {}).items():
        print(f"  {key:12} n={stats['n']:3}  min={stats['min_ms']:6}  "
              f"p50={stats['p50_ms']:6}  max={stats['max_ms']:6}")
    print("\nfloor guidance: --shift-floor-ms ≈ held-key p50 (accept every repeat)"
          "\n                or tap p50 (honor taps, damp autorepeat)")
