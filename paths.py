"""Path resolution for the GeOViLabs web app.

Everything lives in the repository directory. Read-only assets (templates,
static, small models) resolve from the app directory; writable data
(uploads, outputs, frames, tasks.db) resolves from the data directory.
When frozen (PyInstaller), read-only assets are unpacked to sys._MEIPASS
and writable data goes to the environment DATA_DIR (or the current dir).
"""

import os
import sys

BUNDLE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('DATA_DIR') or os.getcwd()

FROZEN = bool(getattr(sys, 'frozen', False))
EXE_DIR = os.path.dirname(sys.executable) if FROZEN else None


def resource(rel):
    """Read-only bundled asset (templates, static, small models)."""
    return os.path.join(BUNDLE_DIR, rel)


def data(rel):
    """Writable location (uploads, outputs, frames, tasks.db)."""
    return os.path.join(DATA_DIR, rel)


def find_model(name):
    """Look for an ONNX model: user's data dir first (big optional models
    like GFPGAN get dropped there), then the bundled static folder."""
    for p in (os.path.join(DATA_DIR, 'models', name),
              resource(os.path.join('static', 'models', name))):
        if os.path.exists(p):
            return p
    return None