"""Server-side restoration tools via ONNX Runtime (CPU):

- denoise: SCUNet (real-world noise), dynamic input size
- deblur:  NAFNet-GoPro (motion blur / camera shake), traced at 512x512
- colorize: DDColor-tiny (chroma prediction in Lab space), fixed 512x512
- inpaint: LaMa (object removal), fixed 512x512, full-res composite

Models download on first use — see model_store.py.
"""

import threading

import numpy as np
from PIL import Image, ImageFilter

import model_store

_sessions = {}
_session_lock = threading.Lock()
_infer_lock = threading.Lock()


def _get_session(key):
    import onnxruntime as ort
    with _session_lock:
        if key not in _sessions:
            ort.set_default_logger_severity(4)
            _sessions[key] = ort.InferenceSession(
                model_store.model_path(key), providers=['CPUExecutionProvider']
            )
        return _sessions[key]


def _run(key, feeds):
    sess = _get_session(key)
    with _infer_lock:
        return sess.run(None, feeds)[0]


# --- Denoise / Deblur (image-to-image, [0,1] float32 NCHW) ---

# NAFNet's export was traced at 512x512 (its attention layer breaks at other
# sizes), so both cleaners run on 512 windows with a 32px overlap that gets
# cropped away — same core-paste approach as the upscaler's tiling.
WIN = 512
PAD = 32
CORE = WIN - 2 * PAD


def _clean_array(key, arr, progress_cb=None):
    """Run SCUNet/NAFNet over an HxWx3 [0,1] array in 512px windows.

    Both models only accept 512x512 input, so ANY dimension shorter than
    the window gets padded first (a wide-but-short image would otherwise
    produce a non-512 window and crash the graph's fixed reshapes)."""
    input_name = _get_session(key).get_inputs()[0].name
    h, w = arr.shape[:2]

    pad_h = max(0, WIN - h)
    pad_w = max(0, WIN - w)
    if pad_h or pad_w:
        # reflect needs pad < dim; fall back to edge for tiny images
        mode = 'reflect' if (pad_h < h and pad_w < w) else 'edge'
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    hp, wp = arr.shape[:2]

    # Exactly one window: single inference
    if hp == WIN and wp == WIN:
        out = _run(key, {input_name: arr.transpose(2, 0, 1)[np.newaxis]})
        if progress_cb:
            progress_cb(1, 1)
        return np.clip(out[0].transpose(1, 2, 0)[:h, :w], 0.0, 1.0)

    out = np.empty_like(arr)
    ys = list(range(0, hp, CORE))
    xs = list(range(0, wp, CORE))
    total = len(ys) * len(xs)
    done = 0
    for sy in ys:
        for sx in xs:
            # Window positioned so the core region is padded on all sides,
            # clamped to the (padded) image bounds — always exactly 512x512
            y0 = max(0, min(sy - PAD, hp - WIN))
            x0 = max(0, min(sx - PAD, wp - WIN))
            win = arr[y0:y0 + WIN, x0:x0 + WIN]
            res = _run(key, {input_name: win.transpose(2, 0, 1)[np.newaxis]})
            res = res[0].transpose(1, 2, 0)

            cy1 = min(sy + CORE, hp)
            cx1 = min(sx + CORE, wp)
            out[sy:cy1, sx:cx1] = res[sy - y0:cy1 - y0, sx - x0:cx1 - x0]

            done += 1
            if progress_cb:
                progress_cb(done, total)
    return np.clip(out[:h, :w], 0.0, 1.0)


def clean_image(mode, in_path, out_path, progress_cb=None):
    """mode: 'denoise' | 'deblur'. Returns (width, height)."""
    img = Image.open(in_path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    out = _clean_array(mode, arr, progress_cb=progress_cb)
    Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(out_path, 'PNG')
    return img.size


# --- Colorize (DDColor: predict ab chroma from L, keep full-res luminance) ---

DD_SIZE = 512


def colorize_image(in_path, out_path, progress_cb=None):
    import cv2
    img = Image.open(in_path).convert('RGB')
    rgb = np.asarray(img, dtype=np.float32) / 255.0

    if progress_cb:
        progress_cb(30, 'Analyzing luminance...')
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., :1]

    # Model input: grayscale rebuilt from L at 512x512, no extra normalization
    l_small = cv2.resize(L, (DD_SIZE, DD_SIZE))[..., np.newaxis]
    gray_lab = np.concatenate(
        [l_small, np.zeros((DD_SIZE, DD_SIZE, 2), np.float32)], axis=-1)
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)

    if progress_cb:
        progress_cb(45, 'Running AI colorization...')
    ab = _run('colorize',
              {_get_session('colorize').get_inputs()[0].name:
               gray_rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32)})
    ab = ab[0].transpose(1, 2, 0)

    if progress_cb:
        progress_cb(85, 'Compositing color...')
    ab_full = cv2.resize(ab, (rgb.shape[1], rgb.shape[0]))
    ab_full = np.clip(ab_full, -127.0, 127.0)
    out = cv2.cvtColor(np.concatenate([L, ab_full], axis=-1), cv2.COLOR_LAB2RGB)
    out = np.clip(out, 0.0, 1.0)
    Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(out_path, 'PNG')
    return img.size


# --- Inpaint (LaMa: fill masked region, composite back at full resolution) ---

LAMA_SIZE = 512


def inpaint_image(in_path, mask_path, out_path, progress_cb=None):
    """mask: white = remove. The model runs at 512x512; only the masked
    region of its output is pasted back onto the full-resolution image."""
    img = Image.open(in_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.NEAREST)

    if progress_cb:
        progress_cb(30, 'Preparing mask...')
    # Slightly grow the mask so edge pixels of the object don't survive
    mask = mask.filter(ImageFilter.MaxFilter(9))

    im512 = np.asarray(img.resize((LAMA_SIZE, LAMA_SIZE), Image.BILINEAR),
                       np.float32) / 255.0
    m512 = np.asarray(mask.resize((LAMA_SIZE, LAMA_SIZE), Image.BILINEAR),
                      np.float32) / 255.0
    m512 = (m512 > 0.5).astype(np.float32)
    if not m512.any():
        img.save(out_path, 'PNG')
        return img.size

    if progress_cb:
        progress_cb(45, 'Running AI object removal...')
    sess = _get_session('erase')
    names = [i.name for i in sess.get_inputs()]
    out = _run('erase', {names[0]: im512.transpose(2, 0, 1)[np.newaxis],
                         names[1]: m512[np.newaxis, np.newaxis]})
    # This export takes [0,1] input but emits [0,255]
    result512 = np.clip(out[0].transpose(1, 2, 0) / 255.0, 0.0, 1.0)

    if progress_cb:
        progress_cb(85, 'Compositing...')
    filled = Image.fromarray((result512 * 255.0 + 0.5).astype(np.uint8)) \
        .resize(img.size, Image.LANCZOS)
    # Feathered mask so the paste blends into the untouched full-res pixels
    alpha = mask.filter(ImageFilter.GaussianBlur(3))
    final = Image.composite(filled, img, alpha)
    final.save(out_path, 'PNG')
    return img.size
