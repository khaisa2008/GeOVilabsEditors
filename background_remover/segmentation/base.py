"""Segmentation engines — common interface + generic ONNX runner."""

import threading

import numpy as np
from PIL import Image

from ..models import get_session
from ..utils import to_pil


class MaskSegmenter:
    """Base class. ``segment(image: PIL.Image) -> float32 0..1 mask``."""

    name = 'base'

    def segment(self, image):
        raise NotImplementedError


class ONNXSegmenter(MaskSegmenter):
    """Runs a generic single-output binary-mask ONNX model.

    Subclasses configure the normalisation style and output handling.
    Intended to be shared by BiRefNet, RMBG-*, U²-Net and similar models.
    """

    # Normalisation presets
    PRESETS = {
        'imagenet': dict(
            mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
            std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
        ),
        'unit': dict(mean=np.zeros(3, dtype=np.float32),
                     std=np.ones(3, dtype=np.float32)),
        'half': dict(mean=np.full(3, 0.5, dtype=np.float32),
                     std=np.ones(3, dtype=np.float32)),
    }

    def __init__(self, model_path, input_size=1024, norm='imagenet'):
        self.model_path = model_path
        self.input_size = input_size
        self.norm_key = norm
        self._session = None
        # RLock: `segment()` holds the lock while it lazily resolves
        # `input_name`/`session`, which acquire the same lock themselves.
        self._lock = threading.RLock()
        self._input_name = None

    @property
    def session(self):
        if self._session is None:
            with self._lock:
                if self._session is None:
                    self._session = get_session(self.model_path)
        return self._session

    @property
    def input_name(self):
        if self._input_name is None:
            with self._lock:
                if self._input_name is None:
                    self._input_name = self.session.get_inputs()[0].name
        return self._input_name

    def _normalize(self, arr):
        cfg = self.PRESETS[self.norm_key]
        return (arr - cfg['mean']) / cfg['std']

    def _postprocess(self, out, size):
        """Convert model output to a full-contrast float mask (0..1)."""
        pred = out[0, 0].astype(np.float32) if out.ndim == 4 \
            else np.squeeze(out).astype(np.float32)
        if pred.min() < -0.01 or pred.max() > 1.01:
            pred = 1.0 / (1.0 + np.exp(-pred))
        # Guarded contrast boost. A naive per-image min-max stretch forces
        # every map to the full [0, 1] range, which amplifies model noise in
        # near-flat regions into opaque patches and widens the edge ramp.
        # Only very flat maps (span < 0.6 between the 0.5th/99.5th
        # percentiles) get a modest boost; calibrated probabilities pass
        # through as-is. The narrower percentiles keep faint but real
        # semi-transparent hair from being washed out.
        lo, hi = np.percentile(pred, 0.5), np.percentile(pred, 99.5)
        span = float(hi - lo)
        if span > 1e-6 and span < 0.6:
            pred = np.clip((pred - lo) / span, 0.0, 1.0)
        mask8 = (np.clip(pred, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        alpha = to_pil(mask8).resize(size, Image.LANCZOS)
        return np.asarray(alpha, dtype=np.float32) / 255.0

    def segment(self, image):
        sess = self.session
        # Model input size (BiRefNet-family uses 1024x1024 by default)
        n = self.input_size
        small = image.convert('RGB').resize((n, n), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0
        arr = self._normalize(arr)
        inp = arr.transpose(2, 0, 1)[np.newaxis]

        with self._lock:
            out = sess.run(None, {self.input_name: np.ascontiguousarray(inp)})[0]
        return self._postprocess(out, image.size)