"""Keyboard-first TUI driver for the transcript-correction workflow — a presentation
driver bound onto `cjm-transcript-correction-core`'s operation vocabulary (the core
stays headless; this library owns interaction only). Born on-graph: every region of
this package is authored as graph nodes and projected to `.py`."""

import os

__version__ = "0.0.2"

# PipeWire routing: conda-forge PortAudio ships an ALSA that cannot see the system
# pipewire/default PCM (hw-only enumeration -> audio lands on the wrong sink, e.g. a
# monitor instead of the earbuds the OS routes to). Point it at the system ALSA config
# HERE, in the package __init__, because canonical emit hoists a module own import
# block above any module-level code — only the package boundary runs strictly before
# a submodule imports sounddevice. Pre-set env always wins (setdefault).
if os.path.exists("/usr/share/alsa/alsa.conf"):
    os.environ.setdefault("ALSA_CONFIG_PATH", "/usr/share/alsa/alsa.conf")
if os.path.isdir("/usr/lib/x86_64-linux-gnu/alsa-lib"):
    os.environ.setdefault("ALSA_PLUGIN_DIR", "/usr/lib/x86_64-linux-gnu/alsa-lib")
