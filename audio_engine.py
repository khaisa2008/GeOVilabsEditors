"""Server-side audio tools via ONNX Runtime (CPU):

- separate: MDX-Net (Kim_Vocal_2) vocal/instrumental stem separation, 44.1kHz
- denoise:  DeepFilterNet3 speech enhancement, 48kHz

Audio IO goes through the bundled ffmpeg (any input format ffmpeg reads).
Models download on first use — see model_store.py.

The DFN3 implementation follows the DSP contract published with the
soniqo/DeepFilterNet3-ONNX export and libDF v0.5.6: vorbis-window STFT with
wnorm analysis scaling, ERB mean-norm and complex unit-norm features
(alpha 0.99), ERB mask + order-5 deep filtering. The one-frame output
offset and mask alignment below were validated empirically (SI-SDR
+5 dB on a 10 dB white-noise test; wrong offsets score negative).
"""

import os
import subprocess
import sys
import threading

import numpy as np

import model_store
import paths

_sessions = {}
_session_lock = threading.Lock()
_infer_lock = threading.Lock()


def _ffmpeg():
    exe = 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg'
    if paths.EXE_DIR:
        p = os.path.join(paths.EXE_DIR, exe)
        if os.path.isfile(p):
            return p
    # Dev mode: prefer the static build vendored for the desktop app —
    # system ffmpeg installs (e.g. Anaconda's) can be dynamic or crippled
    import glob
    for p in sorted(glob.glob(os.path.join(paths.BUNDLE_DIR, 'src-tauri',
                                           'binaries', 'ffmpeg-*'))):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    import shutil
    return shutil.which('ffmpeg') or 'ffmpeg'


def _get_session(key):
    import onnxruntime as ort
    with _session_lock:
        if key not in _sessions:
            ort.set_default_logger_severity(4)
            _sessions[key] = ort.InferenceSession(
                model_store.model_path(key), providers=['CPUExecutionProvider']
            )
        return _sessions[key]


def load_audio(path, sr, channels):
    """Decode any ffmpeg-readable file to float32 (channels, n)."""
    raw = subprocess.run(
        [_ffmpeg(), '-v', 'error', '-i', path, '-f', 'f32le',
         '-ac', str(channels), '-ar', str(sr), '-'],
        capture_output=True)
    if raw.returncode != 0 or not raw.stdout:
        err = raw.stderr.decode(errors='replace').strip().splitlines()
        raise ValueError(err[-1][:200] if err else 'Could not decode audio')
    data = np.frombuffer(raw.stdout, np.float32)
    return data.reshape(-1, channels).T.copy()


def save_audio(path, data, sr):
    """Encode float32 (channels, n) to the format implied by the extension."""
    if data.ndim == 1:
        data = data[np.newaxis]
    subprocess.run(
        [_ffmpeg(), '-y', '-v', 'error', '-f', 'f32le',
         '-ac', str(data.shape[0]), '-ar', str(sr), '-i', '-', path],
        input=np.ascontiguousarray(data.T, np.float32).tobytes(), check=True)


def probe_duration(path):
    """Duration in seconds via ffmpeg (works without ffprobe)."""
    r = subprocess.run([_ffmpeg(), '-hide_banner', '-i', path],
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
    import re
    m = re.search(r'Duration:\s*(\d+):(\d+):([\d.]+)', r.stderr)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


# --- Vocal separation (MDX-Net, Kim_Vocal_2) ---

MDX_SR = 44100
MDX_NFFT = 7680
MDX_HOP = 1024
MDX_DIM_F = 3072
MDX_DIM_T = 256
MDX_SEG = (MDX_DIM_T - 1) * MDX_HOP        # samples per model window (~5.9s)
MDX_OVERLAP = MDX_SR                        # 1s crossfade between windows

_mdx_window = np.hanning(MDX_NFFT + 1)[:-1].astype(np.float32)


def _mdx_stft(x):
    pad = MDX_NFFT // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode='reflect')
    frames = np.lib.stride_tricks.sliding_window_view(
        xp, MDX_NFFT, axis=1)[:, ::MDX_HOP]
    return np.fft.rfft(frames[:, :MDX_DIM_T] * _mdx_window,
                       axis=-1).transpose(0, 2, 1)


def _mdx_istft(spec):
    frames = np.fft.irfft(spec.transpose(0, 2, 1), n=MDX_NFFT,
                          axis=-1) * _mdx_window
    T = spec.shape[2]
    length = (T - 1) * MDX_HOP + MDX_NFFT
    out = np.zeros((2, length), np.float32)
    norm = np.zeros(length, np.float32)
    wsq = _mdx_window ** 2
    for t in range(T):
        out[:, t * MDX_HOP:t * MDX_HOP + MDX_NFFT] += frames[:, t]
        norm[t * MDX_HOP:t * MDX_HOP + MDX_NFFT] += wsq
    out /= np.maximum(norm, 1e-8)
    pad = MDX_NFFT // 2
    return out[:, pad:pad + MDX_SEG]


def separate(in_path, vocals_path, inst_path, progress_cb=None):
    """Split a song into vocals + instrumental. Returns duration seconds."""
    sess = _get_session('separate')
    input_name = sess.get_inputs()[0].name
    mix = load_audio(in_path, MDX_SR, 2)
    n = mix.shape[1]

    vocal = np.zeros_like(mix)
    weight = np.zeros(n, np.float32)
    step = MDX_SEG - MDX_OVERLAP
    positions = list(range(0, n, step))
    for idx, pos in enumerate(positions):
        seg = mix[:, pos:pos + MDX_SEG]
        L = seg.shape[1]
        if L < MDX_SEG:
            seg = np.pad(seg, ((0, 0), (0, MDX_SEG - L)))
        spec = _mdx_stft(seg)
        x = np.stack([spec.real[:, :MDX_DIM_F], spec.imag[:, :MDX_DIM_F]],
                     1).reshape(4, MDX_DIM_F, MDX_DIM_T).astype(np.float32)
        with _infer_lock:
            out = sess.run(None, {input_name: x[np.newaxis]})[0][0]
        vspec = np.zeros_like(spec)
        vspec[:, :MDX_DIM_F] = out[[0, 2]] + 1j * out[[1, 3]]
        v = _mdx_istft(vspec)[:, :L]

        w = np.ones(L, np.float32)
        ramp = min(MDX_OVERLAP, L)
        if pos > 0:
            w[:ramp] = np.linspace(0, 1, ramp)
        if pos + MDX_SEG < n:
            w[-ramp:] = np.minimum(w[-ramp:], np.linspace(1, 0, ramp))
        vocal[:, pos:pos + L] += v * w
        weight[pos:pos + L] += w
        if progress_cb:
            progress_cb(idx + 1, len(positions))

    vocal /= np.maximum(weight, 1e-8)
    save_audio(vocals_path, vocal, MDX_SR)
    save_audio(inst_path, mix - vocal, MDX_SR)
    return n / MDX_SR


# --- Denoise (DeepFilterNet3, official export) ---
#
# Faithful batch-mode port of the official libDF/tract runtime (v0.5.6):
# vorbis-window STFT with wnorm analysis scaling, ERB mean-norm and complex
# unit-norm features (alpha 0.99), encoder -> ERB-mask decoder + deep-filter
# decoder, LSNR-gated stages, mask + order-5 deep filtering. Validated to
# reproduce the official `deep-filter` CLI output at ~77 dB SI-SDR (i.e.
# numerically identical) and its 20.8 dB test-set score.

DF_SR = 48000
DF_FFT = 960
DF_HOP = 480
DF_NB_ERB = 32
DF_NB_DF = 96
DF_ORDER = 5
DF_LOOKAHEAD = 2
DF_WNORM = 1.0 / (DF_FFT * DF_FFT / (2 * DF_HOP))   # libDF analysis scaling
DF_ALPHA = 0.99
# libDF tract defaults: below MIN -> silence, above MAX_ERB -> untouched,
# above MAX_DF -> mask only, else mask + deep filter
DF_MIN_DB, DF_MAX_ERB_DB, DF_MAX_DF_DB = -10.0, 30.0, 20.0
DF_CHUNK = 60 * DF_SR                        # process 60s at a time
DF_CHUNK_OVERLAP = DF_SR                     # 1s crossfade

_df_consts = None


def _erb_widths(sr=DF_SR, fft_size=DF_FFT, nb_bands=DF_NB_ERB, min_nb_freqs=2):
    """Port of libDF erb_fb(): FFT bins per ERB band."""
    freq2erb = lambda f: 9.265 * np.log(1 + f / (24.7 * 9.265))
    erb2freq = lambda e: 24.7 * 9.265 * (np.exp(e / 9.265) - 1)
    freq_width = sr / fft_size
    erb_low, erb_high = freq2erb(0.0), freq2erb(sr / 2)
    step = (erb_high - erb_low) / nb_bands
    widths = []
    prev_freq = 0
    freq_over = 0
    for i in range(1, nb_bands + 1):
        fb = int(round(erb2freq(erb_low + i * step) / freq_width))
        nb = fb - prev_freq - freq_over
        if nb < min_nb_freqs:
            freq_over = min_nb_freqs - nb
            nb = min_nb_freqs
        else:
            freq_over = 0
        widths.append(nb)
        prev_freq = fb
    widths[-1] += 1
    too_large = sum(widths) - (fft_size // 2 + 1)
    if too_large > 0:
        widths[-1] -= too_large
    return widths


def _df_constants():
    """ERB forward/inverse matrices and the vorbis window (libDF formulas)."""
    global _df_consts
    if _df_consts is None:
        widths = _erb_widths()
        nb_freqs = DF_FFT // 2 + 1
        erb_fb = np.zeros((nb_freqs, DF_NB_ERB), np.float32)
        erb_inv = np.zeros((DF_NB_ERB, nb_freqs), np.float32)
        pos = 0
        for b, w in enumerate(widths):
            erb_fb[pos:pos + w, b] = 1.0 / w
            erb_inv[b, pos:pos + w] = 1.0
            pos += w
        n = np.arange(DF_FFT)
        s = np.sin(0.5 * np.pi * (n + 0.5) / (DF_FFT // 2))
        window = np.sin(0.5 * np.pi * s * s).astype(np.float32)
        _df_consts = (erb_fb, erb_inv, window)
    return _df_consts


def _df_sessions():
    import onnxruntime as ort
    with _session_lock:
        if 'dfn3' not in _sessions:
            ort.set_default_logger_severity(4)
            mk = lambda name: ort.InferenceSession(
                paths.find_model(name), providers=['CPUExecutionProvider'])
            _sessions['dfn3'] = (mk('dfn3_enc.onnx'), mk('dfn3_erb_dec.onnx'),
                                 mk('dfn3_df_dec.onnx'))
        return _sessions['dfn3']


def _denoise_chunk(sig):
    """Denoise one mono float32 chunk at 48k with the official DFN3 graphs."""
    enc, erb_dec, df_dec = _df_sessions()
    erb_fb, erb_inv, window = _df_constants()
    n = len(sig)
    # libDF streaming framing (frame t covers [(t-1)*hop, (t+1)*hop)), plus
    # tail room for the model's 2-frame lookahead
    x = np.concatenate([np.zeros(DF_HOP, np.float32), sig,
                        np.zeros(DF_FFT + DF_LOOKAHEAD * DF_HOP, np.float32)])
    nfr = (len(x) - DF_FFT) // DF_HOP + 1
    frames = np.lib.stride_tricks.sliding_window_view(x, DF_FFT)[::DF_HOP][:nfr]
    spec = np.fft.rfft(frames * window, axis=-1) * DF_WNORM

    erb_db = 10 * np.log10((np.abs(spec) ** 2) @ erb_fb + 1e-10)
    m = np.linspace(-60, -90, DF_NB_ERB)
    fe = np.empty_like(erb_db)
    for t in range(nfr):
        m = DF_ALPHA * m + (1 - DF_ALPHA) * erb_db[t]
        fe[t] = (erb_db[t] - m) / 40.0
    s = np.linspace(1e-3, 1e-4, DF_NB_DF)
    fs = np.empty((nfr, DF_NB_DF), np.complex128)
    for t in range(nfr):
        s = DF_ALPHA * s + (1 - DF_ALPHA) * np.abs(spec[t, :DF_NB_DF])
        fs[t] = spec[t, :DF_NB_DF] / np.sqrt(s)

    with _infer_lock:
        e0, e1, e2, e3, emb, c0, lsnr = enc.run(None, {
            'feat_erb': fe[np.newaxis, np.newaxis].astype(np.float32),
            'feat_spec': np.stack([fs.real, fs.imag])[np.newaxis].astype(np.float32),
        })
        mask = erb_dec.run(None, {'emb': emb, 'e3': e3, 'e2': e2,
                                  'e1': e1, 'e0': e0})[0]
        coefs = df_dec.run(None, {'emb': emb, 'c0': c0})[0]
    lsnr = lsnr.reshape(-1)
    mask = mask.reshape(nfr, DF_NB_ERB)
    coefs = coefs.reshape(nfr, DF_NB_DF, DF_ORDER, 2)
    c = coefs[..., 0] + 1j * coefs[..., 1]

    # Network outputs at step k+lookahead serve output frame k
    idx = np.minimum(np.arange(nfr) + DF_LOOKAHEAD, nfr - 1)
    lsnr_k, mask_k, c_k = lsnr[idx], mask[idx], c[idx]

    out = spec * (mask_k @ erb_inv)
    padded = np.pad(spec[:, :DF_NB_DF], ((DF_LOOKAHEAD, DF_LOOKAHEAD), (0, 0)))
    df = np.zeros((nfr, DF_NB_DF), np.complex128)
    for i in range(DF_ORDER):                     # taps k-2 .. k+2
        df += c_k[:, :, i] * padded[i:i + nfr]
    use_df = lsnr_k <= DF_MAX_DF_DB
    out[:, :DF_NB_DF] = np.where(use_df[:, np.newaxis], df, out[:, :DF_NB_DF])
    out[lsnr_k > DF_MAX_ERB_DB] = spec[lsnr_k > DF_MAX_ERB_DB]   # clean: untouched
    out[lsnr_k < DF_MIN_DB] = 0                                   # noise only: mute

    fr = np.fft.irfft(out * DF_FFT, n=DF_FFT, axis=-1) * window
    y = np.zeros(len(x), np.float32)
    for t in range(nfr):
        y[t * DF_HOP:t * DF_HOP + DF_FFT] += fr[t]
    return y[DF_HOP:DF_HOP + n]


def denoise(in_path, out_path, progress_cb=None):
    """Remove background noise from speech. Returns duration seconds."""
    _df_sessions()
    sig = load_audio(in_path, DF_SR, 1)[0]
    n = len(sig)

    out = np.zeros(n, np.float32)
    weight = np.zeros(n, np.float32)
    step = DF_CHUNK - DF_CHUNK_OVERLAP
    positions = list(range(0, n, step))
    for idx, pos in enumerate(positions):
        chunk = sig[pos:pos + DF_CHUNK]
        y = _denoise_chunk(chunk)
        L = len(y)
        w = np.ones(L, np.float32)
        ramp = min(DF_CHUNK_OVERLAP, L)
        if pos > 0:
            w[:ramp] = np.linspace(0, 1, ramp)
        if pos + DF_CHUNK < n:
            w[-ramp:] = np.minimum(w[-ramp:], np.linspace(1, 0, ramp))
        out[pos:pos + L] += y * w
        weight[pos:pos + L] += w
        if progress_cb:
            progress_cb(idx + 1, len(positions))

    out /= np.maximum(weight, 1e-8)
    save_audio(out_path, out, DF_SR)
    return n / DF_SR
