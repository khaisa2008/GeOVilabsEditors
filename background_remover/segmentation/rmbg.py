"""RMBG (Remove Background) family segmentation engines.

RMBG-1.4 / RMBG-2.0 export well to ONNX. No URL is hard-coded here: if a
user drops a compatible ``*.onnx`` file into the data models dir, this
engine can use it; otherwise the pipeline silently falls back to BiRefNet.
"""

import os

from .base import ONNXSegmenter


class RMBGSegmenter(ONNXSegmenter):
    name = 'rmbg'

    def __init__(self, model_path=None, input_size=1024, norm='unit'):
        path = model_path or _find('rmbg*.onnx')
        if not path or not os.path.exists(path):
            raise FileNotFoundError('RMBG model not found — provide a '
                                    'model_path or drop rmbg*.onnx into the models dir')
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