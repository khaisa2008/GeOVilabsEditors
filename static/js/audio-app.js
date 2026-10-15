/**
 * GeOViLabs — Audio tools controllers:
 *   Vocal Remover (MDX-Net on the server), Noise Remover (DeepFilterNet3),
 *   Mastering + Tag Editor (ReMaster core, entirely in this window via
 *   the vendored worker — nothing is uploaded).
 */
(function () {
    'use strict';

    const $ = (sel) => document.querySelector(sel);
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const NGU_TOKEN = document.querySelector('meta[name="ngu-token"]')?.content || '';

    // ========================= SHARED HELPERS =========================

    function showToast(msg) {
        const existing = document.querySelector('.toast');
        if (existing) existing.remove();
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = msg;
        document.body.appendChild(toast);
        requestAnimationFrame(() => toast.classList.add('show'));
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    function desktopDialog() {
        return (window.__TAURI__ && window.__TAURI__.dialog) ? window.__TAURI__.dialog : null;
    }

    async function browserSaveAs(suggestedName, getBlob) {
        if (!window.showSaveFilePicker) return false;
        let handle;
        try {
            handle = await window.showSaveFilePicker({ suggestedName });
        } catch (e) {
            if (e && e.name === 'AbortError') return true;
            return false;
        }
        try {
            const blob = await getBlob();
            const writable = await handle.createWritable();
            await writable.write(blob);
            await writable.close();
            showToast('Saved ' + suggestedName);
        } catch (e) {
            showToast('Save failed: ' + e.message);
        }
        return true;
    }

    async function saveServerResult(taskId, suggestedName) {
        const dlg = desktopDialog();
        if (dlg) {
            try {
                const path = await dlg.save({ defaultPath: suggestedName });
                if (!path) return;
                const res = await fetch('/api/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-NGU-Token': NGU_TOKEN },
                    body: JSON.stringify({ task_id: taskId, path }),
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'save failed');
                showToast('Saved to ' + data.path);
            } catch (e) {
                showToast('Save failed: ' + e.message);
            }
            return;
        }
        const url = `/api/audio/download/${taskId}`;
        if (await browserSaveAs(suggestedName, async () => {
            const r = await fetch(url);
            if (!r.ok) throw new Error('download failed');
            return r.blob();
        })) return;
        const a = document.createElement('a');
        a.href = url;
        a.download = suggestedName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    async function saveBlobResult(blob, suggestedName) {
        const dlg = desktopDialog();
        if (dlg) {
            try {
                const path = await dlg.save({ defaultPath: suggestedName });
                if (!path) return;
                const res = await fetch('/api/save-blob?path=' + encodeURIComponent(path), {
                    method: 'POST',
                    headers: { 'X-NGU-Token': NGU_TOKEN },
                    body: blob,
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'save failed');
                showToast('Saved to ' + data.path);
            } catch (e) {
                showToast('Save failed: ' + e.message);
            }
            return;
        }
        if (await browserSaveAs(suggestedName, () => Promise.resolve(blob))) return;
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = suggestedName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    }

    async function ensureModel(key, onProgress) {
        let s = await (await fetch(`/api/models/${key}`)).json();
        if (s.available) return;
        const mb = Math.round((s.size || 0) / 1e6);
        onProgress(0, `Downloading AI model (${mb} MB) — one-time setup...`);
        await fetch(`/api/models/${key}`, { method: 'POST' });
        while (true) {
            await sleep(1000);
            try {
                s = await (await fetch(`/api/models/${key}`)).json();
            } catch {
                continue;
            }
            if (s.available) return;
            if (s.status === 'error') throw new Error('Model download failed: ' + (s.error || 'network error'));
            onProgress(Math.round((s.progress || 0) * 0.99),
                `Downloading AI model (${mb} MB)... ${s.progress || 0}%`);
        }
    }

    async function uploadAudio(file) {
        const formData = new FormData();
        formData.append('audio', file);
        const res = await fetch('/api/audio/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Upload failed');
        return data;
    }

    function pollStatus(taskId, onProgress, isCancelled) {
        return new Promise((resolve, reject) => {
            const iv = setInterval(async () => {
                if (isCancelled && isCancelled()) {
                    clearInterval(iv);
                    reject(new Error('cancelled'));
                    return;
                }
                try {
                    const s = await (await fetch(`/api/image/status/${taskId}`)).json();
                    if (onProgress) onProgress(s.progress || 50, s.message || 'Processing...');
                    if (s.status === 'completed') { clearInterval(iv); resolve(s); }
                    else if (s.status === 'cancelled') { clearInterval(iv); reject(new Error('cancelled')); }
                    else if (s.status === 'error') { clearInterval(iv); reject(new Error(s.message || 'Server error')); }
                } catch {
                    clearInterval(iv);
                    reject(new Error('Connection lost'));
                }
            }, 700);
        });
    }

    function bindDropzone(zoneSel, inputSel, onFile) {
        const zone = $(zoneSel);
        const input = $(inputSel);
        zone.addEventListener('click', () => input.click());
        zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
        zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) onFile(e.dataTransfer.files[0]);
        });
        input.addEventListener('change', () => {
            if (input.files.length) onFile(input.files[0]);
        });
    }

    function isValidAudio(file) {
        return (file.type.startsWith('audio/') || file.type === 'video/mp4'
            || file.name.match(/\.(mp3|wav|flac|m4a|aac|ogg|opus|wma|aiff?|mp4)$/i))
            && file.size <= 500 * 1024 * 1024;
    }

    function fmtDur(sec) {
        const m = Math.floor(sec / 60), s = Math.round(sec % 60);
        return `${m}:${String(s).padStart(2, '0')}`;
    }

    function fmtSize(bytes) {
        if (bytes > 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        return (bytes / 1024).toFixed(0) + ' KB';
    }

    // ========================= VOCAL REMOVER =========================

    const sep = { file: null, taskId: null, stems: null, processing: false, cancelRequested: false, originalUrl: null };

    function initSep() {
        bindDropzone('#sep-dropzone', '#sep-file-input', (file) => {
            if (!isValidAudio(file)) { showToast('Please select an audio file under 500MB.'); return; }
            sep.file = file;
            sep.taskId = null;
            sep.stems = null;
            $('#sep-file-name').textContent = file.name;
            $('#sep-file-info').textContent = fmtSize(file.size);
            $('#sep-upload-section').classList.add('hidden');
            $('#sep-app-section').classList.remove('hidden');
            $('#sep-new-row').classList.remove('hidden');
            $('#sep-results').classList.add('hidden');
            $('#sep-start-btn').disabled = false;
        });

        $('#sep-start-btn').addEventListener('click', sepStart);
        $('#sep-cancel-btn').addEventListener('click', () => {
            sep.cancelRequested = true;
            if (sep.taskId) fetch(`/api/image/cancel/${sep.taskId}`, { method: 'POST' }).catch(() => {});
        });
        $('#sep-new-btn').addEventListener('click', () => {
            $('#sep-app-section').classList.add('hidden');
            $('#sep-new-row').classList.add('hidden');
            $('#sep-upload-section').classList.remove('hidden');
            $('#sep-file-input').value = '';
        });
        $('#sep-vocals-save').addEventListener('click', () => {
            if (sep.stems) saveServerResult(sep.stems.vocals, stemName('vocals'));
        });
        $('#sep-inst-save').addEventListener('click', () => {
            if (sep.stems) saveServerResult(sep.stems.instrumental, stemName('instrumental'));
        });
        $('#sep-zip-btn').addEventListener('click', sepZip);
    }

    function stemName(kind) {
        const stem = sep.file ? sep.file.name.replace(/\.[^.]+$/, '') : 'song';
        return `${stem}_${kind}.wav`;
    }

    function sepProgress(pct, msg) {
        $('#sep-progress-fill').style.width = pct + '%';
        $('#sep-progress-percent').textContent = pct + '%';
        $('#sep-progress-message').textContent = msg;
    }

    async function sepStart() {
        if (sep.processing || !sep.file) return;
        sep.processing = true;
        sep.cancelRequested = false;
        $('#sep-start-btn').disabled = true;
        $('#sep-progress-section').classList.remove('hidden');
        $('#sep-results').classList.add('hidden');
        $('#sep-new-row').classList.add('hidden');

        try {
            await ensureModel('separate', sepProgress);
            sepProgress(2, 'Uploading audio...');
            const up = await uploadAudio(sep.file);
            sep.taskId = up.task_id;
            $('#sep-file-info').textContent = `${fmtSize(sep.file.size)} • ${fmtDur(up.duration)}`;
            const res = await fetch('/api/audio/separate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: up.task_id }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error || 'Failed to start');
            }
            await pollStatus(up.task_id, sepProgress, () => sep.cancelRequested);
            sep.stems = { vocals: up.task_id + '_voc', instrumental: up.task_id + '_ins' };
            if (sep.originalUrl) URL.revokeObjectURL(sep.originalUrl);
            sep.originalUrl = URL.createObjectURL(sep.file);
            $('#sep-original-audio').src = sep.originalUrl;
            $('#sep-vocals-audio').src = `/api/audio/download/${sep.stems.vocals}`;
            $('#sep-inst-audio').src = `/api/audio/download/${sep.stems.instrumental}`;
            $('#sep-results').classList.remove('hidden');
        } catch (err) {
            if (sep.cancelRequested || (err && err.message === 'cancelled')) {
                showToast('Cancelled.');
            } else {
                showToast('Separation failed: ' + err.message);
            }
        } finally {
            sep.processing = false;
            $('#sep-progress-section').classList.add('hidden');
            $('#sep-new-row').classList.remove('hidden');
            $('#sep-start-btn').disabled = false;
        }
    }

    async function sepZip() {
        if (!sep.stems) return;
        const btn = $('#sep-zip-btn');
        btn.disabled = true;
        btn.textContent = 'Zipping...';
        try {
            const res = await fetch('/api/zip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_ids: [sep.stems.vocals, sep.stems.instrumental] }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Zip failed');
            const stem = sep.file ? sep.file.name.replace(/\.[^.]+$/, '') : 'stems';
            const dlg = desktopDialog();
            if (dlg) {
                await saveServerResult(data.zip_id, `${stem}_stems.zip`);
            } else if (!await browserSaveAs(`${stem}_stems.zip`, async () => {
                const r = await fetch(`/api/zip/${data.zip_id}`);
                if (!r.ok) throw new Error('download failed');
                return r.blob();
            })) {
                const a = document.createElement('a');
                a.href = `/api/zip/${data.zip_id}`;
                a.download = `${stem}_stems.zip`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
        } catch (e) {
            showToast('Zip failed: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Download Both (.zip)';
        }
    }

    // ========================= NOISE REMOVER =========================

    const aden = { file: null, taskId: null, processing: false, cancelRequested: false, originalUrl: null };

    function initAdenoise() {
        bindDropzone('#adenoise-dropzone', '#adenoise-file-input', (file) => {
            if (!isValidAudio(file)) { showToast('Please select an audio file under 500MB.'); return; }
            aden.file = file;
            aden.taskId = null;
            if (aden.originalUrl) URL.revokeObjectURL(aden.originalUrl);
            aden.originalUrl = URL.createObjectURL(file);
            $('#adenoise-file-name').textContent = file.name;
            $('#adenoise-file-info').textContent = fmtSize(file.size);
            $('#adenoise-upload-section').classList.add('hidden');
            $('#adenoise-app-section').classList.remove('hidden');
            $('#adenoise-new-row').classList.remove('hidden');
            $('#adenoise-results').classList.add('hidden');
            $('#adenoise-start-btn').disabled = false;
        });

        $('#adenoise-start-btn').addEventListener('click', adenStart);
        $('#adenoise-cancel-btn').addEventListener('click', () => {
            aden.cancelRequested = true;
            if (aden.taskId) fetch(`/api/image/cancel/${aden.taskId}`, { method: 'POST' }).catch(() => {});
        });
        $('#adenoise-new-btn').addEventListener('click', () => {
            $('#adenoise-app-section').classList.add('hidden');
            $('#adenoise-new-row').classList.add('hidden');
            $('#adenoise-upload-section').classList.remove('hidden');
            $('#adenoise-file-input').value = '';
        });
        $('#adenoise-save').addEventListener('click', () => {
            if (!aden.taskId) return;
            const stem = aden.file ? aden.file.name.replace(/\.[^.]+$/, '') : 'audio';
            saveServerResult(aden.taskId, `${stem}_denoised.wav`);
        });
    }

    function adenProgress(pct, msg) {
        $('#adenoise-progress-fill').style.width = pct + '%';
        $('#adenoise-progress-percent').textContent = pct + '%';
        $('#adenoise-progress-message').textContent = msg;
    }

    async function adenStart() {
        if (aden.processing || !aden.file) return;
        aden.processing = true;
        aden.cancelRequested = false;
        $('#adenoise-start-btn').disabled = true;
        $('#adenoise-progress-section').classList.remove('hidden');
        $('#adenoise-results').classList.add('hidden');
        $('#adenoise-new-row').classList.add('hidden');

        try {
            await ensureModel('denoise_speech', adenProgress);
            adenProgress(2, 'Uploading audio...');
            const up = await uploadAudio(aden.file);
            aden.taskId = up.task_id;
            $('#adenoise-file-info').textContent = `${fmtSize(aden.file.size)} • ${fmtDur(up.duration)}`;
            const res = await fetch('/api/audio/denoise', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: up.task_id }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error || 'Failed to start');
            }
            await pollStatus(up.task_id, adenProgress, () => aden.cancelRequested);
            $('#adenoise-before-audio').src = aden.originalUrl;
            $('#adenoise-after-audio').src = `/api/audio/download/${up.task_id}?t=${Date.now()}`;
            $('#adenoise-results').classList.remove('hidden');
        } catch (err) {
            if (aden.cancelRequested || (err && err.message === 'cancelled')) {
                showToast('Cancelled.');
            } else {
                showToast('Denoise failed: ' + err.message);
            }
        } finally {
            aden.processing = false;
            $('#adenoise-progress-section').classList.add('hidden');
            $('#adenoise-new-row').classList.remove('hidden');
            $('#adenoise-start-btn').disabled = false;
        }
    }

    // ========================= MASTERING (in-browser, ReMaster core) =========================

    const BAND_LABELS = ['20', '60', '150', '400', '1k', '2.5k', '6k', '12k'];
    const master = {
        file: null, worker: null, nextId: 1, audio: null, analysis: null,
        autoPlan: null, result: null, resultBlob: null, processing: false,
        player: null, raf: 0, envA: null, envB: null,
    };
    const CURVE_KEYS = { 'neutral': 'neutral', 'bass-forward': 'bassFwd', 'open / acoustic': 'acoustic', 'warm': 'warm' };

    function masterWorker() {
        if (!master.worker) master.worker = new Worker('/static/js/vendor/remaster-worker.js');
        return master.worker;
    }

    function workerCall(kind, payload, onProgress) {
        return new Promise((resolve, reject) => {
            const id = master.nextId++;
            const w = masterWorker();
            const handler = (e) => {
                const m = e.data;
                if (m.id !== id) return;
                if (m.kind === 'progress') { if (onProgress) onProgress(m.stage, m.fraction); return; }
                w.removeEventListener('message', handler);
                if (m.kind === 'error') reject(new Error(m.message));
                else resolve(m);
            };
            w.addEventListener('message', handler);
            w.postMessage({ id, kind, ...payload });
        });
    }

    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const kv = (k, v) => `<tr><td>${k}</td><td class="num">${v}</td></tr>`;
    const dbfs = (v) => (v <= 0 ? '-inf' : (20 * Math.log10(v)).toFixed(1));
    const fmt1 = (n) => (n >= 0 ? '+' : '') + n.toFixed(1);
    const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

    function deltaCell(before, after, unit, betterUp) {
        const d = after - before;
        const cls = Math.abs(d) < 0.05 ? '' : (d > 0) === betterUp ? 'up' : 'down';
        return `${before.toFixed(1)}${unit} → <b>${after.toFixed(1)}${unit}</b> <span class="delta ${cls}">(${fmt1(d)})</span>`;
    }

    function barChart(barsEl, labelsEl, bands) {
        const finite = bands.filter(Number.isFinite);
        const max = finite.length ? Math.max(...finite) : 0;
        barsEl.innerHTML = bands.map((v) => {
            const rel = Number.isFinite(v) ? Math.max(0, Math.min(1, (v - max + 48) / 48)) : 0;
            return `<i style="height:${(rel * 100).toFixed(1)}%"></i>`;
        }).join('');
        labelsEl.innerHTML = BAND_LABELS.map((l) => `<span>${l}</span>`).join('');
    }

    function initMaster() {
        bindDropzone('#master-dropzone', '#master-file-input', masterLoad);
        $('#master-start-btn').addEventListener('click', masterStart);
        $('#master-new-btn').addEventListener('click', masterReset);
        $('#master-save').addEventListener('click', () => {
            if (!master.resultBlob) return;
            const stem = master.file ? master.file.name.replace(/\.[^.]+$/, '') : 'track';
            saveBlobResult(master.resultBlob, `${stem}_mastered.wav`);
        });
        $('#master-play').addEventListener('click', () => {
            const p = master.player;
            if (!p) return;
            if (p.playing) { p.stop(); $('#master-play').textContent = 'Play'; }
            else { p.play(); $('#master-play').textContent = 'Pause'; masterTick(); }
        });
        $('#master-ab-a').addEventListener('click', () => masterAB('A'));
        $('#master-ab-b').addEventListener('click', () => masterAB('B'));
        $('#master-matched').addEventListener('change', (e) => {
            if (master.player) master.player.setMatched(e.target.checked);
        });
        $('#master-wave').addEventListener('click', (e) => {
            const p = master.player;
            if (!p) return;
            const r = e.currentTarget.getBoundingClientRect();
            p.seek(((e.clientX - r.left) / r.width) * p.duration);
            masterPaint();
        });
        window.addEventListener('resize', () => { master.envA = null; if (master.result) masterPaint(); });

        // Tuning controls: live value labels + reset
        const fmtDb = (v) => `${v >= 0 ? '+' : ''}${Number(v).toFixed(1)} dB`;
        const bind = (id, labelId, fmt) => {
            const el = $(id);
            el.addEventListener('input', () => { $(labelId).textContent = fmt(el.value); });
        };
        bind('#tune-eq', '#tune-eq-val', (v) => `${v}%`);
        bind('#tune-bass', '#tune-bass-val', fmtDb);
        bind('#tune-treble', '#tune-treble-val', fmtDb);
        bind('#tune-deess', '#tune-deess-val', (v) => `${Number(v).toFixed(1)} dB`);
        bind('#tune-width', '#tune-width-val', (v) => `\u00d7${Number(v).toFixed(2)}`);
        bind('#tune-lufs', '#tune-lufs-val', (v) => `${v} LUFS`);
        $('#master-reset-tuning').addEventListener('click', () => {
            if (master.autoPlan) masterSeedTuning(master.autoPlan);
        });
    }

    function masterSeedTuning(plan) {
        const set = (id, labelId, value, fmt) => {
            $(id).value = value;
            if (labelId) $(labelId).textContent = fmt(value);
        };
        $('#tune-curve').value = CURVE_KEYS[plan.curveName] || 'neutral';
        set('#tune-eq', '#tune-eq-val', Math.round(plan.eqStrength * 100), (v) => `${v}%`);
        set('#tune-bass', '#tune-bass-val', plan.lowShelf.db, (v) => `${v >= 0 ? '+' : ''}${Number(v).toFixed(1)} dB`);
        set('#tune-treble', '#tune-treble-val', plan.highShelf.db, (v) => `${v >= 0 ? '+' : ''}${Number(v).toFixed(1)} dB`);
        set('#tune-deess', '#tune-deess-val', Math.round(plan.deEss * 2) / 2, (v) => `${Number(v).toFixed(1)} dB`);
        set('#tune-lufs', '#tune-lufs-val', Math.round(plan.targetLufs), (v) => `${v} LUFS`);
        set('#tune-width', '#tune-width-val', Math.round(plan.width * 20) / 20, (v) => `\u00d7${Number(v).toFixed(2)}`);
    }

    function masterOverrides() {
        return {
            curveKey: $('#tune-curve').value,
            eqStrength: parseFloat($('#tune-eq').value) / 100,
            bassDb: parseFloat($('#tune-bass').value),
            trebleDb: parseFloat($('#tune-treble').value),
            deEss: parseFloat($('#tune-deess').value),
            targetLufs: parseInt($('#tune-lufs').value),
            width: parseFloat($('#tune-width').value),
        };
    }

    async function masterLoad(file) {
        if (!isValidAudio(file)) { showToast('Please select an audio file.'); return; }
        masterStopPlayer();
        master.file = file;
        master.result = null;
        master.resultBlob = null;
        $('#master-file-name').textContent = file.name;
        $('#master-file-info').textContent = fmtSize(file.size) + ' • decoding...';
        $('#master-upload-section').classList.add('hidden');
        $('#master-app-section').classList.remove('hidden');
        $('#master-new-row').classList.remove('hidden');
        $('#master-results').classList.add('hidden');
        $('#master-measured').classList.add('hidden');
        $('#master-tuning').classList.add('hidden');
        $('#master-start-btn').disabled = true;
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const buf = await ctx.decodeAudioData(await file.arrayBuffer());
            ctx.close();
            const channels = [];
            for (let c = 0; c < Math.min(2, buf.numberOfChannels); c++) channels.push(buf.getChannelData(c).slice());
            master.audio = { sampleRate: buf.sampleRate, channels };
            $('#master-file-info').textContent = `${fmtSize(file.size)} • measuring...`;
            const analysed = await workerCall('analyse', { audio: master.audio });
            master.analysis = analysed.analysis;
            masterRenderMeasured(master.analysis);
            const planned = await workerCall('plan', { analysis: master.analysis });
            master.autoPlan = planned.plan;
            masterSeedTuning(master.autoPlan);
            $('#master-tuning').classList.remove('hidden');
            $('#master-file-info').textContent =
                `${fmtSize(file.size)} • ${mmss(buf.duration)} • ${buf.sampleRate} Hz • runs in this window, nothing is uploaded`;
            $('#master-start-btn').disabled = false;
        } catch (e) {
            showToast('Could not decode this audio file.');
            masterReset();
        }
    }

    function masterRenderMeasured(a) {
        const f = a.features || {};
        $('#master-measured-table').innerHTML =
            kv('Integrated loudness', Number.isFinite(a.lufs) ? `${a.lufs.toFixed(1)} LUFS` : 'silent') +
            kv('Loudness range', `${a.lra.toFixed(1)} LU`) +
            kv('True peak', `${dbfs(a.truePeak)} dBTP`) +
            kv('Crest factor', `${a.crest.toFixed(1)} dB`) +
            kv('Stereo correlation', a.correlation.toFixed(2) +
                (a.correlation > 0.98 ? ' (near mono)' : a.correlation < 0 ? ' (out of phase)' : '')) +
            kv('Character', `${Math.round(f.centroid || 0)} Hz centroid · ${(f.onsets || 0).toFixed(1)} onsets/s · ${((f.bassRatio || 0) * 100).toFixed(0)}% low end`) +
            kv('Format', `${a.channels} ch · ${a.sampleRate} Hz · ${mmss(a.duration)}`);
        barChart($('#master-bars-before'), $('#master-barlabels-before'), a.bands);
        const hot = a.lufs > -9 && a.crest < 8;
        $('#master-hot').classList.toggle('hidden', !hot);
        if (hot) $('#master-hot').textContent =
            `Already at ${a.lufs.toFixed(1)} LUFS with only ${a.crest.toFixed(1)} dB of crest factor — heavily limited already. There is little headroom to work with; master from a quieter mix if you have one.`;
        $('#master-measured').classList.remove('hidden');
    }

    function masterProgress(pct, msg) {
        $('#master-progress-fill').style.width = pct + '%';
        $('#master-progress-percent').textContent = pct + '%';
        $('#master-progress-message').textContent = msg;
    }

    async function masterStart() {
        if (master.processing || !master.audio || !master.analysis) return;
        master.processing = true;
        masterStopPlayer();
        $('#master-start-btn').disabled = true;
        $('#master-progress-section').classList.remove('hidden');
        $('#master-results').classList.add('hidden');
        try {
            masterProgress(3, 'Starting...');
            const m = await workerCall('master', { audio: master.audio, analysis: master.analysis, overrides: masterOverrides() },
                (stage, fraction) => masterProgress(Math.round(3 + fraction * 95), stage));
            master.result = m;
            const wav = window.RemasterLib.encodeWav24(m.result.audio);
            master.resultBlob = new Blob([wav], { type: 'audio/wav' });
            // Unhide first: the waveform canvas needs a real clientWidth to paint
            $('#master-results').classList.remove('hidden');
            masterRenderResult(master.analysis, m.result, m.plan, m.after);
        } catch (err) {
            showToast('Mastering failed: ' + err.message);
        } finally {
            master.processing = false;
            $('#master-progress-section').classList.add('hidden');
            $('#master-start-btn').disabled = false;
        }
    }

    function masterRenderResult(before, r, plan, after) {
        $('#master-why-list').innerHTML = plan.why.map((w) => `<div class="reason">${esc(w)}</div>`).join('');
        $('#master-plan-table').innerHTML =
            kv('Spectral target', esc(plan.curveName)) +
            kv('Loudness target', r.backedOff
                ? `<span class="warn">${r.actualTarget} LUFS — backed off ${r.backedOff} dB from ${plan.targetLufs}, reaching it would have meant audible clipping</span>`
                : `${plan.targetLufs} LUFS · ceiling ${plan.ceilingDb} dBTP`) +
            kv('De-essing', plan.deEss > 0.2 ? `up to ${plan.deEss.toFixed(1)} dB` : 'not needed') +
            kv('Stereo width', plan.width === 1 ? 'left as recorded' : `×${plan.width.toFixed(2)}`);
        const fromDb = window.RemasterLib.fromDb;
        $('#master-delta-table').innerHTML =
            kv('Integrated loudness', deltaCell(before.lufs, after.lufs, ' LUFS', true)) +
            kv('True peak', deltaCell(20 * Math.log10(before.truePeak + 1e-12), 20 * Math.log10(after.truePeak + 1e-12), ' dBTP', false)) +
            kv('Crest factor', deltaCell(before.crest, after.crest, ' dB', true)) +
            kv('Corrective EQ', r.corrections.length
                ? r.corrections.map((c) => `${c.label}Hz ${c.db >= 0 ? '+' : ''}${c.db.toFixed(1)}`).join(' · ')
                : 'balance already on target') +
            kv('Gain applied', `${fmt1(r.gainDb)} dB`) +
            kv('Peak limiting', r.maxGainReduction < 0.05 ? 'none needed' : `${r.maxGainReduction.toFixed(1)} dB max reduction`) +
            kv('Soft clipping', r.clipped < 1e-6 ? 'no peaks reached the knee'
                : `<span class="${r.clipped > 0.005 ? 'warn' : ''}">${(r.clipped * 100).toFixed(3)}% of samples rounded${r.clipped > 0.005 ? ' — this source resists loudness' : ''}</span>`) +
            kv('Delivery check', after.truePeak <= fromDb(plan.ceilingDb) + 1e-4
                ? `<span class="good">within ${plan.ceilingDb} dBTP ceiling</span>` : '<span class="warn">over ceiling</span>');
        barChart($('#master-bars-after'), $('#master-barlabels-after'), after.bands);

        // A/B player, loudness matched
        const p = new window.RemasterLib.ABPlayer();
        p.load(master.audio, r.audio, before.lufs, after.lufs);
        p.onEnded = () => { $('#master-play').textContent = 'Play'; masterPaint(); };
        master.player = p;
        masterAB('B');
        $('#master-matched').checked = true;
        $('#master-play').textContent = 'Play';
        const trimDb = -20 * Math.log10(Math.min(p.matchA, p.matchB));
        const quieter = p.matchB < 1 ? 'mastered version' : 'original';
        $('#master-note').textContent = trimDb > 0.05
            ? `Matched playback attenuates the ${quieter} by ${trimDb.toFixed(1)} dB so both play at the same loudness — otherwise the louder one wins on level alone. Untick to hear the real delivered difference.`
            : 'Both versions are already at the same loudness.';
        master.envA = null;
        masterPaint();
        $('#master-time').textContent = `0:00 / ${mmss(p.duration)}`;
    }

    function masterAB(which) {
        if (master.player) master.player.set(which);
        $('#master-ab-a').classList.toggle('on', which === 'A');
        $('#master-ab-b').classList.toggle('on', which === 'B');
    }

    function masterPaint() {
        const canvas = $('#master-wave');
        if (!master.result || !master.player) return;
        const w = Math.max(200, Math.floor(canvas.clientWidth) || 800);
        if (!master.envA || master.envA.length !== w) {
            master.envA = window.RemasterLib.envelope(master.audio, w);
            master.envB = window.RemasterLib.envelope(master.result.result.audio, w);
        }
        const p = master.player;
        drawWave(canvas, master.envA, master.envB, p.playing ? p.pos() / p.duration : (p.offset > 0 ? p.offset / p.duration : -1));
    }

    // Mirrored halves: original on top, mastered below (dark-theme colours)
    function drawWave(canvas, envA, envB, posFrac) {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth, h = canvas.clientHeight;
        canvas.width = w * dpr; canvas.height = h * dpr;
        const g = canvas.getContext('2d');
        g.scale(dpr, dpr);
        g.clearRect(0, 0, w, h);
        const mid = h / 2;
        g.fillStyle = 'rgba(148, 163, 184, 0.55)';
        for (let x = 0; x < envA.length; x++) { const v = Math.min(1, envA[x]) * mid; g.fillRect(x, mid - v, 1, Math.max(1, v)); }
        g.fillStyle = 'rgba(52, 230, 161, 0.85)';
        for (let x = 0; x < envB.length; x++) { const v = Math.min(1, envB[x]) * mid; g.fillRect(x, mid, 1, Math.max(1, v)); }
        g.strokeStyle = '#2a5540'; g.lineWidth = 1;
        g.beginPath(); g.moveTo(0, mid + 0.5); g.lineTo(w, mid + 0.5); g.stroke();
        g.font = '10px Inter, sans-serif'; g.textBaseline = 'top';
        g.fillStyle = '#94a3b8'; g.fillText('ORIGINAL', 6, 5);
        g.fillStyle = '#34e6a1'; g.fillText('MASTERED', 6, mid + 5);
        if (posFrac >= 0) {
            g.strokeStyle = '#00d084'; g.lineWidth = 1.5;
            g.beginPath(); g.moveTo(posFrac * w, 0); g.lineTo(posFrac * w, h); g.stroke();
        }
    }

    function masterTick() {
        const p = master.player;
        if (!p) return;
        $('#master-time').textContent = `${mmss(p.pos())} / ${mmss(p.duration)}`;
        masterPaint();
        if (!p.playing) { $('#master-play').textContent = 'Play'; return; }
        master.raf = requestAnimationFrame(masterTick);
    }

    function masterStopPlayer() {
        cancelAnimationFrame(master.raf);
        if (master.player) { master.player.dispose(); master.player = null; }
    }

    function masterReset() {
        masterStopPlayer();
        master.audio = null;
        master.analysis = null;
        master.result = null;
        $('#master-app-section').classList.add('hidden');
        $('#master-new-row').classList.add('hidden');
        $('#master-results').classList.add('hidden');
        $('#master-measured').classList.add('hidden');
        $('#master-tuning').classList.add('hidden');
        $('#master-upload-section').classList.remove('hidden');
        $('#master-file-input').value = '';
    }

    // ========================= TAG EDITOR (in-browser, ReMaster core) =========================

    const TAG_FIELDS = ['title', 'artist', 'album', 'year', 'genre', 'track', 'comment'];
    const tags = { file: null, bytes: null, resultBlob: null, resultExt: '' };

    function initTags() {
        bindDropzone('#tags-dropzone', '#tags-file-input', async (file) => {
            if (!file.name.match(/\.(mp3|wav)$/i)) {
                showToast('The Tag Editor supports MP3 and WAV files.');
                return;
            }
            tags.file = file;
            tags.resultBlob = null;
            tags.bytes = new Uint8Array(await file.arrayBuffer());
            $('#tags-file-name').textContent = file.name;
            $('#tags-file-info').textContent = fmtSize(file.size);
            $('#tags-upload-section').classList.add('hidden');
            $('#tags-app-section').classList.remove('hidden');
            $('#tags-new-row').classList.remove('hidden');
            $('#tags-results').classList.add('hidden');
            $('#tags-clean-btn').disabled = false;
            // Read-only preview: a strip run in memory just to list what is there
            try {
                const r = await workerCall('tags', { bytes: tags.bytes, mode: 'strip', tags: {} });
                renderTagTable($('#tags-before-table'), $('#tags-before-title'), 'Tags in this file', r.result.tagsBefore);
            } catch { /* table stays empty */ }
        });

        document.querySelectorAll('input[name="tags-mode"]').forEach((r) => {
            r.addEventListener('change', () => {
                $('#tags-fields').classList.toggle('hidden', tagsMode() !== 'write');
            });
        });
        $('#tags-clean-btn').addEventListener('click', tagsProcess);
        $('#tags-new-btn').addEventListener('click', () => {
            $('#tags-app-section').classList.add('hidden');
            $('#tags-new-row').classList.add('hidden');
            $('#tags-upload-section').classList.remove('hidden');
            $('#tags-file-input').value = '';
            tags.bytes = null;
        });
        $('#tags-save').addEventListener('click', () => {
            if (!tags.resultBlob) return;
            const stem = tags.file ? tags.file.name.replace(/\.[^.]+$/, '') : 'audio';
            saveBlobResult(tags.resultBlob, `${stem}_clean${tags.resultExt}`);
        });
    }

    const tagsMode = () => document.querySelector('input[name="tags-mode"]:checked').value;

    function tagFields() {
        const t = {};
        for (const f of TAG_FIELDS) t[f] = (document.querySelector(`[data-tagfield="${f}"]`).value || '').trim();
        return t;
    }

    function renderTagTable(table, titleEl, title, rows) {
        if (titleEl) titleEl.textContent = rows && rows.length ? `${title} — ${rows.length} field${rows.length > 1 ? 's' : ''}` : title;
        table.innerHTML = '';
        if (!rows || !rows.length) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = 2;
            td.textContent = 'No tags present.';
            tr.appendChild(td);
            table.appendChild(tr);
            return;
        }
        for (const row of rows) {
            const tr = document.createElement('tr');
            const k = document.createElement('td');
            k.textContent = row.label;
            const v = document.createElement('td');
            const s = String(row.value).replace(/\s*\r?\n\s*/g, ' · ');
            v.textContent = s.length > 300 ? `${s.slice(0, 300)}… (${s.length} chars)` : s;
            tr.append(k, v);
            table.appendChild(tr);
        }
    }

    async function tagsProcess() {
        if (!tags.bytes) return;
        $('#tags-clean-btn').disabled = true;
        try {
            const mode = tagsMode();
            const r = await workerCall('tags', { bytes: tags.bytes, mode, tags: mode === 'write' ? tagFields() : {} });
            const res = r.result;
            if (res.error) throw new Error(res.error);
            tags.resultBlob = new Blob([res.bytes], { type: res.mime || 'application/octet-stream' });
            tags.resultExt = res.ext ? '.' + res.ext.replace(/^\./, '') : '';
            const total = res.inputLength || 1;
            const seg = (part, cls) => part ? `<div class="${cls}" style="flex:${Math.max((part / total) * 100, 1.5)} 0 auto"></div>` : '';
            $('#tags-map').innerHTML = seg(res.front, 'cut') + seg(res.payloadLength, 'keep') + seg(res.back, 'cut');
            const delta = res.bytes.length - res.inputLength;
            $('#tags-result-table').innerHTML =
                kv('Tags found', res.names.length ? esc(res.names.join(', ')) : 'none found') +
                kv('Audio payload', `${res.payloadLength.toLocaleString()} bytes`) +
                kv('Verdict', res.audioIdentical
                    ? '<span class="good">audio byte-for-byte identical — lossless</span>'
                    : '<span class="warn">MISMATCH — do not use</span>') +
                kv('File size', `${res.inputLength.toLocaleString()} → ${res.bytes.length.toLocaleString()} B (${delta >= 0 ? '+' : ''}${delta.toLocaleString()})`) +
                kv('Written back', mode === 'write' ? 'your tags' : 'nothing — every tag removed');
            renderTagTable($('#tags-after-table'), $('#tags-after-title'), 'Tags after', res.tagsAfter);
            $('#tags-results').classList.remove('hidden');
        } catch (e) {
            showToast('Tag processing failed: ' + e.message);
        } finally {
            $('#tags-clean-btn').disabled = false;
        }
    }

    // The mastering worker doubles as the tag worker
    initSep();
    initAdenoise();
    initMaster();
    initTags();
})();
