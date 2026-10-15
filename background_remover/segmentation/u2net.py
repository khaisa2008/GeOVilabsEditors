"""U²-Net segmentation engine (ONNX).

U²-Net is a lightweight saliency model commonly exported to ONNX at
320x320. Provided as a drop-in modular option; not bundled by default.
"""

import os

from .base import ONNXSegmenter


class U2NetSegmenter(ONNXSegmenter):
    name = 'u2net'

    def __init__(self, model_path=None, input_size=320, norm='imagenet'):
        path = model_path or _find('u2net*.onnx')
        if not path or not os.path.exists(path):
            raise FileNotFoundError('U2-Net model not found — provide a '
                                    'model_path or drop u2net*.onnx into the models dir')
        super().__init__(path, input_size=input_size, norm=norm)


def _find(pattern):
    import glob
    try:
        import paths
    except Exception:
        return None
    for base in (paths.data('models'),
                 paths.resource(os.path.join('static', 'models'))):
        if not base or not os.path.isdir(base):
            continue
        hits = glob.glob(os.path.join(base, pattern))
        if hits:
            return hits[0]
    return None