/**
 * GeOViLabs — Background remover controller.
 * BiRefNet runs on the server; the model downloads once on first use.
 * Single mode: one image, pick a background, compare panels.
 * Batch mode: multiple images, per-item progress, zip download.
 */
(function () {
    'use strict';

    const bgState = {
        taskId: null,
        file: null,
        originalUrl: null,
        background: 'transparent',
        method: 'ai',        // 'ai' | 'color'
        removeColor: '#ffffff',
        tolerance: 30,
        picking: false,
        processing: false,
        cancelRequested: false,
    };

    const bgBatch = {
        items: [],       // { file, url, status, taskId, el }
        running: false,
        background: 'transparent',
        method: 'ai',
        removeColor: '#ffffff',
        tolerance: 30,
        cancelRequested: false,
        currentTaskId: null,
    };

    const $ = (sel) => document.querySelector(sel);
    const NGU_TOKEN = document.querySelector('meta[name="ngu-token"]')?.content || '';
    const setupSection = $('#bg-setup-section');
    const uploadSection = $('#bg-upload-section');
    const appSection = $('#bg-app-section');
    const batchSection = $('#bg-batch-section');
    const dropzone = $('#bg-dropzone');
    const fileInput = $('#bg-file-input');
    const originalImg = $('#bg-original');
    const resultImg = $('#bg-result');
    const outputPlaceholder = $('#bg-output-placeholder');
    const startBtn = $('#bg-start-btn');
    const progressSection = $('#bg-progress-section');
    const progressFill = $('#bg-progress-fill');
    const progressMessage = $('#bg-progress-message');
    const progressPercent = $('#bg-progress-percent');
    const downloadSection = $('#bg-download-section');
    const newRow = $('#bg-new-row');

    let bgCompare = null;

    init();

    async function init() {
        bindEvents();
        if (window.GeoviImg) {
            bgCompare = window.GeoviImg.initCompare({
                section: '#bg-compare-section',
                box: '#bg-compare-box',
                before: '#bg-compare-before',
                after: '#bg-compare-after',
                beforeWrap: '#bg-compare-before-wrap',
                handle: '#bg-compare-handle',
            });
            window.GeoviImg.bindPaste(
                () => !uploadSection.classList.contains('hidden'),
                (files) => handleFiles(files),
                dropzone
            );
        }
        try {
            const s = await (await fetch('/api/bg/model')).json();
            showPanel(s.available);
            if (s.status === 'downloading') resumeModelUI();
        } catch {
            showPanel(false);
        }
    }

    function showPanel(modelReady) {
        setupSection.classList.toggle('hidden', modelReady);
        uploadSection.classList.toggle('hidden', !modelReady);
    }

    function bindEvents() {
        $('#bg-model-btn').addEventListener('click', startModelDownload);

        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) handleFiles(fileInput.files);
        });

        bindChoiceButtons('#bg-choice-buttons', '[data-bgchoice]', '#bg-custom-color-options',
            '#bg-custom-color', bgState,
            (v) => { bgState.background = v; });
        bindChoiceButtons('#bg-batch-choice-buttons', '[data-bgbatchchoice]', '#bg-batch-custom-color-options',
            '#bg-batch-custom-color', bgBatch,
            (v) => { bgBatch.background = v; });

        bindMethodButtons('#bg-method-buttons', '[data-bgmethod]', '#bg-color-options',
            (v) => { bgState.method = v; });
        bindMethodButtons('#bg-batch-method-buttons', '[data-bgbatchmethod]', '#bg-batch-color-options',
            (v) => { bgBatch.method = v; });

        const colorPickers = [
            ['#bg-remove-color', (v) => { bgState.removeColor = v; }],
            ['#bg-batch-remove-color', (v) => { bgBatch.removeColor = v; }],
        ];
        colorPickers.forEach(([sel, cb]) => $(sel).addEventListener('input', (e) => cb(e.target.value)));

        bindTolerance('#bg-tolerance', '#bg-tolerance-val', (v) => { bgState.tolerance = v; });
        bindTolerance('#bg-batch-tolerance', '#bg-batch-tolerance-val', (v) => { bgBatch.tolerance = v; });

        $('#bg-pick-btn').addEventListener('click', togglePicking);
        originalImg.addEventListener('click', (e) => sampleColor(e));

        startBtn.addEventListener('click', startRemoval);
        $('#bg-cancel-btn').addEventListener('click', cancelJob);
        $('#bg-download-btn').addEventListener('click', downloadResult);
        $('#bg-new-btn').addEventListener('click', resetSingle);

        $('#bg-batch-start-btn').addEventListener('click', startBatch);
        $('#bg-batch-zip-btn').addEventListener('click', downloadBatchZip);
        $('#bg-batch-reset-btn').addEventListener('click', resetBatch);
    }

    // Background selector (transparent / hex colour / blur). 'custom' reveals
    // a colour picker whose current value becomes the ``background`` hex
    // sent to the server.
    function bindChoiceButtons(groupSel, btnSel, optsSel, inputSel, target, onChange) {
        const opts = $(optsSel);
        const input = $(inputSel);
        const buttons = document.querySelectorAll(groupSel + ' ' + btnSel);
        let customBtn = null;
        buttons.forEach((btn) => {
            if ((btn.dataset.bgchoice || btn.dataset.bgbatchchoice) === 'custom') customBtn = btn;
            btn.addEventListener('click', () => {
                buttons.forEach((b) => b.classList.remove('selected'));
                btn.classList.add('selected');
                const choice = btn.dataset.bgchoice || btn.dataset.bgbatchchoice;
                const isCustom = choice === 'custom';
                if (opts) opts.classList.toggle('hidden', !isCustom);
                onChange(isCustom ? input.value : choice);
            });
        });
        if (input) {
            input.addEventListener('input', () => {
                if (customBtn && customBtn.classList.contains('selected')) onChange(input.value);
            });
        }
    }
    function bindMethodButtons(groupSel, btnSel, optionsSel, onChange) {
        const buttons = document.querySelectorAll(groupSel + ' ' + btnSel);
        buttons.forEach((btn) => {
            btn.addEventListener('click', () => {
                buttons.forEach((b) => b.classList.remove('selected'));
                btn.classList.add('selected');
                onChange(btn.dataset.bgmethod || btn.dataset.bgbatchmethod);
                $(optionsSel).classList.toggle('hidden', btn.dataset.bgmethod !== 'color'
                    && btn.dataset.bgbatchmethod !== 'color');
            });
        });
    }

    function bindTolerance(rangeSel, valSel, onChange) {
        $(rangeSel).addEventListener('input', (e) => {
            const v = e.target.value;
            $(valSel).textContent = v;
            onChange(parseInt(v, 10));
        });
    }

    function togglePicking() {
        bgState.picking = !bgState.picking;
        const btn = $('#bg-pick-btn');
        btn.classList.toggle('selected', bgState.picking);
        btn.textContent = bgState.picking ? 'Click on the image...' : 'Pick from image';
        originalImg.classList.toggle('color-sampling', bgState.picking);
        if (!bgState.picking) showToast('Color sampling cancelled.');
    }

    // Sample the color under the cursor from the original image.
    function sampleColor(e) {
        if (!bgState.picking || !bgState.file) return;
        const c = document.createElement('canvas');
        const scale = Math.min(1, 2048 / originalImg.naturalWidth);
        c.width = Math.max(1, Math.round(originalImg.naturalWidth * scale));
        c.height = Math.max(1, Math.round(originalImg.naturalHeight * scale));
        const ctx = c.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(originalImg, 0, 0, c.width, c.height);

        const rect = originalImg.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / rect.width * c.width);
        const y = Math.floor((e.clientY - rect.top) / rect.height * c.height);
        if (x < 0 || y < 0 || x >= c.width || y >= c.height) return;
        const [r, g, b] = ctx.getImageData(x, y, 1, 1).data;
        const hex = '#' + [r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('');

        bgState.removeColor = hex;
        $('#bg-remove-color').value = hex;
        bgState.picking = false;
        $('#bg-pick-btn').classList.remove('selected');
        $('#bg-pick-btn').textContent = 'Pick from image';
        originalImg.classList.remove('color-sampling');
        showToast('Picked ' + hex + ' — adjust tolerance if needed.');
    }

    // --- Model download ---

    async function startModelDownload() {
        try {
            await fetch('/api/bg/model', { method: 'POST' });
            resumeModelUI();
        } catch {
            showToast('Could not start the model download — are you online?');
        }
    }

    function resumeModelUI() {
        const btn = $('#bg-model-btn');
        const track = $('#bg-model-track');
        const fill = $('#bg-model-fill');
        const status = $('#bg-model-status');
        btn.disabled = true;
        btn.textContent = 'Downloading...';
        track.classList.remove('hidden');

        const iv = setInterval(async () => {
            try {
                const s = await (await fetch('/api/bg/model')).json();
                if (s.available) {
                    clearInterval(iv);
                    status.textContent = '';
                    showToast('Model installed — background removal is ready.');
                    showPanel(true);
                    return;
                }
                if (s.status === 'error') {
                    clearInterval(iv);
                    btn.disabled = false;
                    btn.textContent = 'Retry Download';
                    status.textContent = 'Download failed: ' + (s.error || 'unknown error');
                    return;
                }
                fill.style.width = (s.progress || 0) + '%';
                status.textContent = `Downloading model... ${s.progress || 0}%`;
            } catch { /* transient — keep polling */ }
        }, 1000);
    }

    // --- File intake ---

    function isValidImage(file) {
        return (file.type.startsWith('image/') || file.name.match(/\.(png|jpe?g|webp|bmp|tiff?)$/i))
            && file.size <= 50 * 1024 * 1024;
    }

    function handleFiles(fileList) {
        const files = Array.from(fileList).filter(isValidImage);
        if (!files.length) {
            showToast('Please select image files under 50MB.');
            return;
        }
        if (files.length === 1) enterSingle(files[0]);
        else enterBatch(files);
    }

    // --- Single image ---

    function enterSingle(file) {
        bgState.file = file;
        bgState.taskId = null;
        if (bgState.originalUrl) URL.revokeObjectURL(bgState.originalUrl);
        bgState.originalUrl = URL.createObjectURL(file);
        originalImg.src = bgState.originalUrl;
        originalImg.onload = () => {
            $('#bg-original-res').textContent = `${originalImg.naturalWidth}x${originalImg.naturalHeight}`;
        };

        uploadSection.classList.add('hidden');
        batchSection.classList.add('hidden');
        appSection.classList.remove('hidden');
        newRow.classList.remove('hidden');
        downloadSection.classList.add('hidden');
        resultImg.classList.add('hidden');
        resultImg.src = '';
        outputPlaceholder.classList.remove('hidden');
        startBtn.disabled = false;
    }

    async function startRemoval() {
        if (bgState.processing || !bgState.file) return;
        bgState.processing = true;
        bgState.cancelRequested = false;
        startBtn.disabled = true;
        progressSection.classList.remove('hidden');
        downloadSection.classList.add('hidden');
        newRow.classList.add('hidden');

        try {
            updateProgress(5, 'Uploading image...');
            const taskId = await uploadAndRemove(bgState.file, {
                background: bgState.background,
                method: bgState.method,
                removeColor: bgState.removeColor,
                tolerance: bgState.tolerance,
            },
                (pct, msg) => updateProgress(pct, msg),
                () => bgState.cancelRequested,
                (id) => { bgState.taskId = id; },
                (meta) => { bgState.meta = meta; });

            const url = `/api/image/download/${taskId}?t=${Date.now()}`;
            resultImg.src = url;
            resultImg.classList.remove('hidden');
            outputPlaceholder.classList.add('hidden');
            resultImg.onload = () => {
                $('#bg-result-res').textContent = `${resultImg.naturalWidth}x${resultImg.naturalHeight}`;
                if (bgCompare) bgCompare.show(bgState.originalUrl, url);
            };
            const m = bgState.meta || {};
            const bgB = bgState.background;
            const bgLabel = bgB === 'transparent' ? 'transparent'
                : bgB === 'blur' ? 'blurred background'
                : /^#[0-9a-fA-F]{6}$/.test(bgB) ? `custom colour ${bgB}`
                : bgB;
            const typeLabel = m.image_type ? m.image_type.charAt(0).toUpperCase() + m.image_type.slice(1) : '';
            const qualityStr = m.quality_score != null ? ` · quality ${Number(m.quality_score).toFixed(2)}` : '';
            const timeStr = m.processing_time != null ? ` · ${Number(m.processing_time).toFixed(1)}s` : '';
            const metaLine = typeLabel ? `${typeLabel}${qualityStr}${timeStr}` : '';
            $('#bg-download-details').textContent = metaLine || (bgState.method === 'color'
                ? 'Transparent PNG by color, ready for design work'
                : `PNG on ${bgLabel}`);

            bgState.processing = false;
            progressSection.classList.add('hidden');
            downloadSection.classList.remove('hidden');
            newRow.classList.remove('hidden');
            startBtn.disabled = false;
        } catch (err) {
            bgState.processing = false;
            progressSection.classList.add('hidden');
            newRow.classList.remove('hidden');
            startBtn.disabled = false;
            if (bgState.cancelRequested || (err && err.message === 'cancelled')) {
                showToast('Cancelled — adjust and try again.');
            } else {
                showToast('Background removal failed: ' + err.message);
            }
        }
    }

    // Upload + start + poll. Shared by single and batch paths.
    async function uploadAndRemove(file, opts, onProgress, isCancelled, onTaskId, onMeta) {
        const formData = new FormData();
        formData.append('image', file);
        const res = await fetch('/api/image/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Upload failed');
        if (onTaskId) onTaskId(data.task_id);
        if (isCancelled && isCancelled()) throw new Error('cancelled');

        const rmRes = await fetch('/api/bg/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_id: data.task_id,
                background: opts.background,
                method: opts.method,
                color: opts.removeColor,
                tolerance: opts.tolerance,
                shadow: opts.shadow || false,
            }),
        });
        if (!rmRes.ok) {
            const err = await rmRes.json().catch(() => ({}));
            throw new Error(err.error || 'Failed to start');
        }

        await new Promise((resolve, reject) => {
            const iv = setInterval(async () => {
                if (isCancelled && isCancelled()) {
                    clearInterval(iv);
                    reject(new Error('cancelled'));
                    return;
                }
                try {
                    const s = await (await fetch(`/api/image/status/${data.task_id}`)).json();
                    if (onProgress) onProgress(s.progress || 50, s.message || 'Processing...');
                    if (s.status === 'completed') {
                        clearInterval(iv);
                        if (onMeta) onMeta({ image_type: s.image_type, quality_score: s.quality_score,
                            confidence: s.confidence, passes: s.passes, processing_time: s.processing_time,
                            method: s.method });
                        resolve();
                    } else if (s.status === 'cancelled') { clearInterval(iv); reject(new Error('cancelled')); }
                    else if (s.status === 'error') { clearInterval(iv); reject(new Error(s.message || 'Server error')); }
                } catch {
                    clearInterval(iv);
                    reject(new Error('Connection lost'));
                }
            }, 500);
        });
        return data.task_id;
    }

    function updateProgress(pct, msg) {
        progressFill.style.width = pct + '%';
        progressPercent.textContent = pct + '%';
        progressMessage.textContent = msg;
    }

    function cancelJob() {
        if (!bgState.processing) return;
        bgState.cancelRequested = true;
        if (bgState.taskId) {
            fetch(`/api/image/cancel/${bgState.taskId}`, { method: 'POST' }).catch(() => {});
        }
    }

    function downloadResult() {
        if (!bgState.taskId) return;
        const stem = bgState.file ? bgState.file.name.replace(/\.[^.]+$/, '') : 'image';
        saveServerResult(bgState.taskId, `${stem}_nobg.png`);
    }

    function resetSingle() {
        bgState.taskId = null;
        bgState.file = null;
        bgState.processing = false;
        bgState.picking = false;
        appSection.classList.add('hidden');
        progressSection.classList.add('hidden');
        downloadSection.classList.add('hidden');
        newRow.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        originalImg.src = '';
        resultImg.src = '';
        resultImg.classList.add('hidden');
        outputPlaceholder.classList.remove('hidden');
        if (bgCompare) bgCompare.reset();
        fileInput.value = '';
        originalImg.classList.remove('color-sampling');
        const pickBtn = $('#bg-pick-btn');
        if (pickBtn) { pickBtn.classList.remove('selected'); pickBtn.textContent = 'Pick from image'; }
    }

    // --- Batch ---

    function enterBatch(files) {
        bgBatch.items.forEach((i) => { if (i.url) URL.revokeObjectURL(i.url); });
        bgBatch.items = files.map((file) => ({
            file, url: URL.createObjectURL(file), status: 'pending', taskId: null, el: null,
        }));
        bgBatch.running = false;
        bgBatch.cancelRequested = false;

        uploadSection.classList.add('hidden');
        appSection.classList.add('hidden');
        batchSection.classList.remove('hidden');

        $('#bg-batch-title').textContent = `Batch: ${bgBatch.items.length} images`;
        $('#bg-batch-sub').textContent = 'Pick a background, then start';
        $('#bg-batch-progress-label').textContent = '—';
        $('#bg-batch-zip-btn').classList.add('hidden');
        $('#bg-batch-start-btn').disabled = false;
        $('#bg-batch-reset-btn').textContent = 'Cancel';

        const grid = $('#bg-batch-grid');
        grid.innerHTML = '';
        bgBatch.items.forEach((item) => {
            const div = document.createElement('div');
            div.className = 'batch-item';
            const img = document.createElement('img');
            img.src = item.url;
            const info = document.createElement('div');
            info.className = 'batch-item-info';
            const name = document.createElement('span');
            name.className = 'batch-item-name';
            name.textContent = item.file.name;
            const status = document.createElement('span');
            status.className = 'batch-status pending';
            status.textContent = 'queued';
            info.append(name, status);
            div.append(img, info);
            grid.appendChild(div);
            item.el = { status, info, img };
        });
    }

    function setItemStatus(item, cls, text) {
        item.el.status.className = 'batch-status ' + cls;
        item.el.status.textContent = text;
    }

    async function startBatch() {
        if (bgBatch.running || !bgBatch.items.length) return;
        bgBatch.running = true;
        bgBatch.cancelRequested = false;
        $('#bg-batch-start-btn').disabled = true;
        $('#bg-batch-zip-btn').classList.add('hidden');
        $('#bg-batch-reset-btn').textContent = 'Cancel';
        $('#bg-batch-sub').textContent = 'Removing backgrounds on the server';

        let done = 0;
        for (const item of bgBatch.items) {
            if (bgBatch.cancelRequested) break;
            $('#bg-batch-progress-label').textContent = `${done + 1} / ${bgBatch.items.length}`;
            setItemStatus(item, 'working', '0%');
            try {
                const taskId = await uploadAndRemove(item.file, {
                    background: bgBatch.background,
                    method: bgBatch.method,
                    removeColor: bgBatch.removeColor,
                    tolerance: bgBatch.tolerance,
                },
                    (pct) => setItemStatus(item, 'working', pct + '%'),
                    () => bgBatch.cancelRequested,
                    (id) => { bgBatch.currentTaskId = id; });
                item.taskId = taskId;
                item.status = 'completed';
                bgBatch.currentTaskId = null;
                setItemStatus(item, 'done', 'done');
                addItemSaveLink(item);
                item.el.img.src = `/api/image/download/${taskId}?t=${Date.now()}`;
            } catch (e) {
                if (bgBatch.cancelRequested || (e && e.message === 'cancelled')) {
                    setItemStatus(item, 'failed', 'cancelled');
                    break;
                }
                console.error('bg batch item failed', e);
                setItemStatus(item, 'failed', 'failed');
            }
            done++;
        }

        const finished = bgBatch.items.filter((i) => i.status === 'completed');
        if (bgBatch.cancelRequested) {
            $('#bg-batch-progress-label').textContent = 'Cancelled';
            $('#bg-batch-sub').textContent = `Cancelled — ${finished.length} finished before stopping`;
        } else {
            $('#bg-batch-progress-label').textContent = 'Complete';
            $('#bg-batch-sub').textContent = `${finished.length} of ${bgBatch.items.length} succeeded`;
        }
        if (finished.length) $('#bg-batch-zip-btn').classList.remove('hidden');

        bgBatch.running = false;
        bgBatch.currentTaskId = null;
        bgBatch.cancelRequested = false;
        $('#bg-batch-start-btn').disabled = false;
        $('#bg-batch-reset-btn').textContent = 'Close';
    }

    function addItemSaveLink(item) {
        const a = document.createElement('a');
        a.textContent = 'save';
        a.href = '#';
        const name = item.file.name.replace(/\.[^.]+$/, '') + '_nobg.png';
        a.addEventListener('click', (e) => {
            e.preventDefault();
            saveServerResult(item.taskId, name);
        });
        item.el.info.appendChild(a);
    }

    async function downloadBatchZip() {
        const ids = bgBatch.items.filter((i) => i.status === 'completed').map((i) => i.taskId);
        if (!ids.length) return;
        const btn = $('#bg-batch-zip-btn');
        btn.disabled = true;
        btn.textContent = 'Zipping...';
        try {
            const res = await fetch('/api/zip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_ids: ids }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Zip failed');
            await saveServerResult(data.zip_id, 'nobg_images.zip', `/api/zip/${data.zip_id}`);
        } catch (e) {
            showToast('Zip failed: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Download All (.zip)';
        }
    }

    function resetBatch() {
        if (bgBatch.running) {
            bgBatch.cancelRequested = true;
            if (bgBatch.currentTaskId) {
                fetch(`/api/image/cancel/${bgBatch.currentTaskId}`, { method: 'POST' }).catch(() => {});
            }
            $('#bg-batch-sub').textContent = 'Cancelling...';
            return;
        }
        bgBatch.items.forEach((i) => { if (i.url) URL.revokeObjectURL(i.url); });
        bgBatch.items = [];
        $('#bg-batch-grid').innerHTML = '';
        batchSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        fileInput.value = '';
    }

    // --- Saving (same logic as everywhere else: native dialog on desktop,
    // "Save As" dialog in Chrome/Edge, normal download elsewhere) ---

    function desktopDialog() {
        return (window.__TAURI__ && window.__TAURI__.dialog) ? window.__TAURI__.dialog : null;
    }

    async function browserSaveAs(suggestedName, getBlob) {
        if (!window.showSaveFilePicker) return false;
        let handle;
        try {
            handle = await window.showSaveFilePicker({ suggestedName });
        } catch (e) {
            if (e && e.name === 'AbortError') return true; // user closed the dialog
            return false; // API blocked here — use the fallback
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

    async function saveServerResult(taskId, suggestedName, browserUrl) {
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
        const url = browserUrl || `/api/image/download/${taskId}`;
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
})();
