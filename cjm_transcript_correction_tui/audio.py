import time
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


def load_chunk(
    wav_path: str,       # The model-input WAV (16 kHz mono) the chunk lives in
    start_s: float,      # Chunk start (seconds, local to this WAV)
    end_s: float,        # Chunk end (seconds, local to this WAV)
    speed: float = 1.0,  # Playback rate (1.0 = natural; offline linear resample)
) -> np.ndarray:  # float32 mono samples ready for `ChunkPlayer.play`
    """Read one VAD chunk's samples from the model-input WAV — frame-sliced, sample-accurate.

    The correction loop verifies against WHAT THE MODEL HEARD, so the slice always
    comes from the model-input rendition, never the source media. Speed control is an
    offline resample of the chunk (near-free at 16 kHz mono), so the persistent output
    stream never changes rate."""
    with sf.SoundFile(wav_path) as f:
        start = max(0, int(round(start_s * f.samplerate)))
        stop = min(len(f), int(round(end_s * f.samplerate)))
        f.seek(start)
        data = f.read(max(0, stop - start), dtype="float32", always_2d=False)
    if data.ndim > 1:  # defensive: model input is mono, but never trust a file
        data = data.mean(axis=1)
    if speed != 1.0 and len(data):
        n = max(1, int(round(len(data) / speed)))
        data = np.interp(np.linspace(0.0, len(data) - 1.0, n),
                         np.arange(len(data)), data).astype(np.float32)
    return data


class ChunkPlayer:
    """Persistent-output-stream VAD-chunk player — the focus-walk auto-play engine.

    ONE `sounddevice.OutputStream` stays open for the player's lifetime; playing a
    chunk swaps the buffer reference the callback consumes (start latency ~= one
    block, ~8 ms at the default 128-frame block @ 16 kHz) and `stop` clears it.
    This is the native inversion of the FastHTML-era Web Audio workaround: PCM is
    handed straight to the device, sample-accurate by construction. The callback
    runs on the PortAudio thread; `play`/`stop` are plain reference swaps, so the
    worst race is a single stale/silent block — inaudible, and the churn-tolerant
    contract the correction loop already accepted (immediate-play-on-focus-change
    ships first; burst suppression is the named refinement, layered ABOVE this
    class off focus-settle, input-agnostically)."""

    def __init__(
        self,
        samplerate: int = 16000,  # Model-input WAV rate (the stream matches the file)
        blocksize: int = 128,     # Frames per callback block (the latency floor)
    ):
        self._buf: Optional[np.ndarray] = None   # Current chunk (float32 mono); None = silence
        self._pos = 0                            # Next frame within _buf
        self._first_block_at: Optional[float] = None  # monotonic ts when the current chunk first sounded
        self.samplerate = samplerate
        try:
            self._stream = sd.OutputStream(
                samplerate=samplerate, channels=1, dtype="float32",
                blocksize=blocksize, callback=self._callback)
            self.stream_rate = samplerate
        except sd.PortAudioError:
            # The device dictates the stream rate, not the file: HDMI-class sinks
            # reject 16 kHz outright — open at the device's preferred rate and
            # resample each chunk on play() instead.
            self.stream_rate = int(sd.query_devices(kind="output")["default_samplerate"])
            self._stream = sd.OutputStream(
                samplerate=self.stream_rate, channels=1, dtype="float32",
                blocksize=blocksize, callback=self._callback)
        self._stream.start()

    def _callback(self, out, frames, time_info, status) -> None:  # PortAudio pull (audio thread)
        buf, pos = self._buf, self._pos
        if buf is None or pos >= len(buf):
            out.fill(0.0)
            return
        if self._first_block_at is None:
            self._first_block_at = time.monotonic()
        n = min(frames, len(buf) - pos)
        out[:n, 0] = buf[pos:pos + n]
        if n < frames:
            out[n:, :].fill(0.0)
        self._pos = pos + n

    def play(self, samples: np.ndarray) -> None:
        """Swap in a chunk (auto-play on focus change) — replaces whatever is sounding.

        `samples` arrive at the model-input rate (`samplerate`); when the device
        forced a different stream rate, the chunk is resampled here (a few ms even
        for a full ~20 s chunk — offline, so the stream itself never changes rate)."""
        if self.stream_rate != self.samplerate and len(samples):
            n = max(1, int(round(len(samples) * self.stream_rate / self.samplerate)))
            samples = np.interp(np.linspace(0.0, len(samples) - 1.0, n),
                                np.arange(len(samples)), samples).astype(np.float32)
        self._first_block_at = None
        self._pos = 0
        self._buf = samples

    def stop(self) -> None:
        """Silence immediately (focus left mid-chunk, Esc, burst traversal)."""
        self._buf = None
        self._pos = 0

    @property
    def playing(self) -> bool:  # True while the current chunk still has frames to sound
        buf = self._buf
        return buf is not None and self._pos < len(buf)

    @property
    def first_block_at(self) -> Optional[float]:  # monotonic ts of the current chunk's first audible block
        return self._first_block_at

    def close(self) -> None:
        """Tear down the stream (app exit)."""
        self._stream.stop()
        self._stream.close()
