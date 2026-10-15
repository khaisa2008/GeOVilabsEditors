/**
 * GeOViLabs — Image Converter controller.
 *
 * Flow:
 *   1. Pick the source + target format (and quality) and press OK.
 *   2. Paste / drag & drop / browse one or many images.
 *   3. A single list shows one card per image, each with its own
 *      queued → converting → converted | error state machine, its own
 *      Convert / Retry / Download button and an honest *estimated*
 *      progress bar. The server converts in one opaque request with no
 *      streaming progress, so the bar eases toward ~90% while the fetch is
 *      in flight and only completes on a real successful response.
 *   4. "Convert All" runs a bounded (3 at a time) worker pool over every
 *      image that is not converted yet. Once everything is converted the
 *      same button becomes "Download All" and zips the results (JSZip).
 *      A selection of converted files can also be zipped on its own.
 *
 * Conversion runs server-side so every format Pillow supports both ways is
 * available: PNG, JPG, WebP, AVIF, BMP, TIFF, GIF, ICO, PDF and SVG/AI.
 * Downloads go straight to the browser's download folder — no save dialog.
 */
(function () {
    'use strict';

    const $ = (sel) => document.querySelector(sel);

    const MAX_BYTES = 50 * 1024 * 1024;
    const CONCURRENCY = 3;
    const ZIP_ALL_NAME = 'geovilabs-converted-images.zip';
    const ZIP_SELECTED_NAME = 'geovilabs-converted-selected.zip';

    const FORMATS = {
        png:  { label: 'PNG',  ext: 'png',  lossy: false },
        jpg:  { label: 'JPG',  ext: 'jpg',  lossy: true },
        webp: { label: 'WebP', ext: 'webp', lossy: true },
        avif: { label: 'AVIF', ext: 'avif', lossy: true },
        bmp:  { label: 'BMP',  ext: 'bmp',  lossy: false },
        tiff: { label: 'TIFF', ext: 'tiff', lossy: false },
        gif:  { label: 'GIF',  ext: 'gif',  lossy: false },
        ico:  { label: 'ICO',  ext: 'ico',  lossy: false },
        pdf:  { label: 'PDF',  ext: 'pdf',  lossy: false },
        svg:  { label: 'SVG',  ext: 'svg',  lossy: false },
        ai:   { label: 'AI',   ext: 'ai',   lossy: false },
    };

    const MIME_TO_FORMAT = {
        'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp',
        'image/avif': 'avif', 'image/bmp': 'bmp', 'image/x-ms-bmp': 'bmp',
        'image/tiff': 'tiff', 'image/gif': 'gif', 'image/x-icon': 'ico',
        'image/vnd.microsoft.icon': 'ico',
    };

    const state = {
        from: 'auto',
        to: 'jpg',
        quality: 90,
        items: [],
        list: null,
        uid: 0,
    };

    let bulkRunning = false;
    let zipping = false;

    function isBusy() {
        return bulkRunning || zipping;
    }

    // ========================= HELPERS =========================

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

    function isKnownImage(file) {
        if (file.type && file.type.indexOf('image/') === 0) return true;
        return /\.(png|jpe?g|webp|avif|bmp|tiff?|gif|ico|heic|heif)$/i.test(file.name || '');
    }

    function detectFormat(file) {
        if (MIME_TO_FORMAT[file.type]) return MIME_TO_FORMAT[file.type];
        const m = (file.name || '').match(/\.([a-z0-9]+)$/i);
        if (!m) return null;
        const ext = m[1].toLowerCase();
        const map = { jpeg: 'jpg', tif: 'tiff' };
        return FORMATS[ext] ? ext : (map[ext] || null);
    }

    function fmtBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    }

    function stemOf(name) {
        return (name || 'image').replace(/\.[^.]+$/, '') || 'image';
    }

    // Straight to the download folder — no save picker/dialog.
    function downloadBlob(blob, filename) {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    }

    function uniqueName(name, used) {
        if (!used.has(name)) {
            used.add(name);
            return name;
        }
        const dot = name.lastIndexOf('.');
        const stem = dot > 0 ? name.slice(0, dot) : name;
        const ext = dot > 0 ? name.slice(dot) : '';
        let n = 1;
        let candidate = stem + '_' + n + ext;
        while (used.has(candidate)) {
            n += 1;
            candidate = stem + '_' + n + ext;
        }
        used.add(candidate);
        return candidate;
    }

    function bindDropzone(zoneSel, inputSel, onFiles) {
        const zone = $(zoneSel);
        const input = $(inputSel);
        if (!zone || !input) return;
        zone.addEventListener('click', () => input.click());
        zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
        zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files);
        });
        input.addEventListener('change', () => {
            if (input.files.length) onFiles(input.files);
            input.value = '';
        });
        if (window.GeoviImg && window.GeoviImg.bindPaste) {
            window.GeoviImg.bindPaste(
                () => {
                    const tab = zone.closest('main.tab-content');
                    const section = zone.closest('section');
                    return !!tab && tab.classList.contains('active') &&
                        !!section && !section.classList.contains('hidden');
                },
                (files) => onFiles(files),
                zone
            );
        }
    }

    // ========================= SCREENS =========================

    function updateQualityVisibility() {
        const group = $('#convert-quality-group');
        if (group) {
            const fmt = FORMATS[state.to];
            group.classList.toggle('hidden', !(fmt && fmt.lossy));
        }
    }

    function updateTargetBadge() {
        const badge = $('#convert-target-badge');
        if (badge) {
            const from = state.from === 'auto' ? 'Auto' : (FORMATS[state.from] ? FORMATS[state.from].label : 'Auto');
            const to = targetLabel();
            badge.textContent = from + ' \u2192 ' + to;
        }
        updateHeadStats();
    }

    function showSetup() {
        const setup = $('#convert-setup-section');
        const upload = $('#convert-upload-section');
        if (setup) setup.classList.remove('hidden');
        if (upload) upload.classList.add('hidden');
    }

    function showUpload() {
        const setup = $('#convert-setup-section');
        const upload = $('#convert-upload-section');
        if (setup) setup.classList.add('hidden');
        if (upload) upload.classList.remove('hidden');
        updateTargetBadge();
        refreshChrome();
    }

    // ========================= CARD RENDERING =========================

    function origLabel(item) {
        return (item.origFormat && FORMATS[item.origFormat]) ? FORMATS[item.origFormat].label : 'IMG';
    }

    function targetLabel() {
        return FORMATS[state.to] ? FORMATS[state.to].label : String(state.to).toUpperCase();
    }

    function convLabel(item) {
        return origLabel(item) + ' \u2192 ' + targetLabel();
    }

    // Rough size estimate for the "Ready" state (server does the real work).
    function estimateSize(item) {
        const factor = {
            jpg: 0.18, avif: 0.09, webp: 0.14, png: 0.85, bmp: 1.2,
            tiff: 1.6, gif: 0.5, ico: 0.05, pdf: 0.55, svg: 0.1, ai: 0.55,
        }[state.to] || 0.3;
        return fmtBytes(Math.max(1024, Math.round(item.file.size * factor)));
    }

    // Signed result: "-87%" when smaller, "+12%" when bigger, null when same.
    function pctSaved(item) {
        if (!item.blob || !item.file.size) return null;
        const pct = Math.round((1 - item.blob.size / item.file.size) * 100);
        if (pct === 0) return null;
        return pct > 0 ? '-' + pct + '%' : '+' + Math.abs(pct) + '%';
    }

    function removeButton(label) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'convert-remove-btn';
        b.setAttribute('aria-label', label || 'Remove');
        b.textContent = '\u00d7';
        return b;
    }

    function buildCard(item) {
        const root = document.createElement('div');
        root.className = 'convert-item';
        root.dataset.id = item.id;
        root.dataset.status = 'queued';

        const main = document.createElement('div');
        main.className = 'convert-item-main';

        const left = document.createElement('div');
        left.className = 'convert-item-left';

        const checkLabel = document.createElement('label');
        checkLabel.className = 'convert-item-check';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'convert-item-checkbox';
        checkbox.setAttribute('aria-label', 'Select ' + item.file.name);
        checkbox.checked = !!item.selected;
        checkLabel.appendChild(checkbox);

        const thumb = document.createElement('div');
        thumb.className = 'convert-item-thumb';
        const img = document.createElement('img');
        img.alt = '';
        img.addEventListener('load', () => {
            if (item.width == null && img.naturalWidth) {
                item.width = img.naturalWidth;
                item.height = img.naturalHeight;
                thumb.title = img.naturalWidth + ' \u00d7 ' + img.naturalHeight;
            }
        });
        img.src = item.url;
        thumb.appendChild(img);

        const thumbBadge = document.createElement('span');
        thumbBadge.className = 'convert-item-thumb-badge';
        thumbBadge.textContent = origLabel(item);
        thumb.appendChild(thumbBadge);

        const info = document.createElement('div');
        info.className = 'convert-item-info';

        const namerow = document.createElement('div');
        namerow.className = 'convert-item-namerow';
        const name = document.createElement('span');
        name.className = 'convert-item-name';
        name.title = item.file.name;
        name.textContent = item.file.name;
        const badge = document.createElement('span');
        badge.className = 'convert-item-badge is-ready';
        badge.textContent = 'Ready';
        namerow.append(name, badge);

        const meta = document.createElement('div');
        meta.className = 'convert-item-meta';
        const metaSize = document.createElement('span');
        metaSize.className = 'convert-meta-size';
        metaSize.textContent = fmtBytes(item.file.size);
        const dot1 = document.createElement('span');
        dot1.className = 'convert-meta-dot';
        dot1.textContent = '\u2022';
        const metaConv = document.createElement('span');
        metaConv.className = 'convert-meta-conv';
        metaConv.textContent = convLabel(item);
        const dot2 = document.createElement('span');
        dot2.className = 'convert-meta-dot';
        dot2.textContent = '\u2022';
        const metaResult = document.createElement('span');
        metaResult.className = 'convert-meta-result is-ready';
        metaResult.textContent = 'Ready';
        meta.append(metaSize, dot1, metaConv, dot2, metaResult);

        info.append(namerow, meta);
        left.append(checkLabel, thumb, info);

        const action = document.createElement('div');
        action.className = 'convert-item-action';

        main.append(left, action);
        root.append(main);

        item.el = {
            root, checkbox, name, thumb, thumbBadge, badge,
            metaSize, metaConv, metaResult, action,
            fill: null, pct: null,
        };
        return root;
    }

    function actionButton(className, label) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'convert-btn ' + className;
        b.textContent = label;
        return b;
    }

    function buildProgress() {
        const wrap = document.createElement('div');
        wrap.className = 'convert-progress';
        const head = document.createElement('div');
        head.className = 'convert-progress-head';
        const pct = document.createElement('span');
        pct.className = 'convert-progress-pct';
        pct.textContent = '0%';
        const label = document.createElement('span');
        label.className = 'convert-progress-label';
        label.textContent = 'Compressing\u2026';
        head.append(pct, label);
        const track = document.createElement('div');
        track.className = 'convert-progress-track';
        const fill = document.createElement('div');
        fill.className = 'convert-progress-fill';
        track.appendChild(fill);
        wrap.append(head, track);
        return { wrap, fill, pct };
    }

    // Applies item.status to the card: root state classes, status badge, meta
    // result text and the action area (buttons or progress). Never invents a
    // "converted" state.
    function setItemStatus(item, status) {
        item.status = status;
        const el = item.el;
        if (!el) return;

        el.root.classList.toggle('is-converting', status === 'converting');
        el.root.classList.toggle('is-converted', status === 'converted');
        el.root.classList.toggle('is-error', status === 'error');
        el.root.dataset.status = status;

        el.action.textContent = '';
        el.fill = null;
        el.pct = null;

        if (status === 'converting') {
            el.badge.className = 'convert-item-badge is-converting';
            el.badge.innerHTML = '<span class="convert-badge-spinner"></span>Converting\u2026';
            el.metaResult.className = 'convert-meta-result is-converting';
            el.metaResult.textContent = 'Processing\u2026';

            const progress = buildProgress();
            el.fill = progress.fill;
            el.pct = progress.pct;
            el.action.append(progress.wrap, removeButton('Cancel conversion'));
            setProgress(item, item.progress || 0);
            return;
        }

        if (status === 'converted') {
            const sizeTxt = item.blob ? fmtBytes(item.blob.size) : '';
            const pct = pctSaved(item);
            el.badge.className = 'convert-item-badge is-done';
            el.badge.textContent = '\u2713 Done';
            el.metaResult.className = 'convert-meta-result is-done';
            el.metaResult.textContent = sizeTxt + (pct ? ' (' + pct + ')' : '');
            el.action.appendChild(actionButton('convert-btn-download', sizeTxt ? 'Download (' + sizeTxt + ')' : 'Download'));
            el.action.appendChild(removeButton());
        } else if (status === 'error') {
            el.badge.className = 'convert-item-badge is-error';
            el.badge.textContent = 'Failed';
            el.metaResult.className = 'convert-meta-result is-error';
            el.metaResult.textContent = item.error || 'Conversion failed';
            el.action.appendChild(actionButton('convert-btn-retry', 'Retry'));
            el.action.appendChild(removeButton());
        } else {
            const cancelled = status === 'cancelled';
            el.badge.className = 'convert-item-badge is-ready';
            el.badge.textContent = cancelled ? 'Cancelled' : 'Ready';
            el.metaResult.className = 'convert-meta-result is-ready';
            el.metaResult.textContent = cancelled ? 'Cancelled' : ('Est. ' + estimateSize(item));
            el.action.appendChild(actionButton('convert-btn-primary', 'Convert'));
            el.action.appendChild(removeButton());
        }
    }

    function setProgress(item, pct) {
        const clamped = Math.max(0, Math.min(100, Math.round(pct)));
        item.progress = clamped;
        if (item.el) {
            if (item.el.fill) item.el.fill.style.width = clamped + '%';
            if (item.el.pct) item.el.pct.textContent = clamped + '%';
        }
    }

    // ========================= ESTIMATED PROGRESS =========================

    function stopEstimate(item) {
        if (item.timer) {
            clearInterval(item.timer);
            item.timer = null;
        }
    }

    function startEstimate(item) {
        stopEstimate(item);
        setProgress(item, 4);
        item.timer = setInterval(() => {
            if (item.status !== 'converting') {
                stopEstimate(item);
                return;
            }
            const remaining = 90 - item.progress;
            if (remaining <= 0) {
                setProgress(item, 90);
                return;
            }
            const step = Math.max(0.4, remaining * 0.06);
            setProgress(item, Math.min(90, item.progress + step));
        }, 260);
    }

    // ========================= ADD / REMOVE =========================

    function addFiles(fileList) {
        const files = Array.from(fileList || []);
        if (!files.length) return;

        const valid = [];
        let notImage = 0;
        let tooBig = 0;

        files.forEach((file) => {
            if (!isKnownImage(file)) { notImage += 1; return; }
            if (file.size > MAX_BYTES) { tooBig += 1; return; }
            valid.push(file);
        });

        const notices = [];
        if (notImage) notices.push(notImage + (notImage > 1 ? ' files' : ' file') + ' skipped (not an image)');
        if (tooBig) notices.push(tooBig + (tooBig > 1 ? ' files' : ' file') + ' skipped (over 50MB)');
        if (notices.length) showToast(notices.join(' \u00b7 ') + '.');

        if (!valid.length) return;

        valid.forEach(addItem);
        showUpload();
        refreshChrome();
    }

    function addItem(file) {
        const item = {
            id: 'ci-' + (++state.uid),
            file: file,
            url: URL.createObjectURL(file),
            origFormat: detectFormat(file),
            width: null,
            height: null,
            status: 'queued',
            progress: 0,
            error: null,
            blob: null,
            filename: null,
            outFormat: null,
            selected: false,
            token: 0,
            timer: null,
            el: null,
        };
        state.items.push(item);

        const card = buildCard(item);
        if (state.list) state.list.appendChild(card);
        setItemStatus(item, 'queued');
    }

    function findItem(id) {
        for (let i = 0; i < state.items.length; i++) {
            if (state.items[i].id === id) return state.items[i];
        }
        return null;
    }

    function disposeItem(item) {
        item.token += 1; // invalidate any in-flight request + late response
        stopEstimate(item);
        if (item.url) {
            URL.revokeObjectURL(item.url);
            item.url = null;
        }
        item.blob = null;
        if (item.el && item.el.root && item.el.root.parentNode) {
            item.el.root.parentNode.removeChild(item.el.root);
        }
    }

    function removeItem(id) {
        const idx = state.items.findIndex((i) => i.id === id);
        if (idx < 0) return;
        disposeItem(state.items[idx]);
        state.items.splice(idx, 1);
        refreshChrome();
    }

    function clearAll() {
        state.items.forEach(disposeItem);
        state.items = [];
        const input = $('#convert-file-input');
        if (input) input.value = '';
        refreshChrome();
    }

    // ========================= CONVERSION =========================

    async function convertOne(item) {
        if (!item || item.status === 'converting') return;

        const token = item.token + 1;
        item.token = token;

        const targetKey = state.to;
        const target = FORMATS[targetKey];
        if (!target) {
            item.error = 'Unsupported output format';
            setItemStatus(item, 'error');
            return;
        }

        item.error = null;
        item.blob = null;
        item.filename = null;
        item.outFormat = targetKey;
        if (item.el && item.el.metaConv) item.el.metaConv.textContent = convLabel(item);

        setItemStatus(item, 'converting');
        startEstimate(item);

        const formData = new FormData();
        formData.append('image', item.file, item.file.name);
        formData.append('format', targetKey);
        formData.append('quality', String(state.quality));

        try {
            const res = await fetch('/api/convert', { method: 'POST', body: formData });
            if (token !== item.token) return; // removed / superseded while in flight

            if (!res.ok) {
                let msg = 'Conversion failed';
                try {
                    const data = await res.json();
                    if (data && data.error) msg = data.error;
                } catch (e) { /* keep default */ }
                throw new Error(msg);
            }

            const blob = await res.blob();
            if (token !== item.token) return;

            // Target changed while we were converting → discard the stale result.
            if (state.to !== targetKey) {
                stopEstimate(item);
                setProgress(item, 0);
                setItemStatus(item, 'queued');
                return;
            }

            item.blob = blob;
            item.filename = stemOf(item.file.name) + '.' + target.ext;
            stopEstimate(item);
            setProgress(item, 100);
            setItemStatus(item, 'converted');
        } catch (err) {
            if (token !== item.token) return;
            stopEstimate(item);
            item.error = (err && err.message) || 'Conversion failed';
            setItemStatus(item, 'error');
        } finally {
            if (token === item.token) stopEstimate(item);
            refreshChrome();
        }
    }

    function downloadItem(item) {
        if (!item || !item.blob) return;
        const target = FORMATS[item.outFormat] || FORMATS[state.to];
        const name = item.filename || (stemOf(item.file.name) + '.' + (target ? target.ext : 'png'));
        downloadBlob(item.blob, name);
    }

    // ========================= BULK CONVERT =========================

    async function convertAll() {
        if (bulkRunning) return;
        if (!state.items.length) return;

        const pending = state.items.filter(
            (i) => i.status !== 'converted' && i.status !== 'converting'
        );
        if (!pending.length) {
            refreshChrome();
            return;
        }

        bulkRunning = true;
        updateMainButton();

        const totalToDo = pending.length;
        let finished = 0;
        let cursor = 0;

        setStatusText('Converting 0 of ' + totalToDo + '\u2026');

        async function worker() {
            while (cursor < pending.length) {
                const item = pending[cursor];
                cursor += 1;

                if (!state.items.includes(item)) { finished += 1; continue; }
                if (item.status === 'converted' || item.status === 'converting') {
                    finished += 1;
                    continue;
                }

                await convertOne(item);
                finished += 1;
                setStatusText('Converting ' + Math.min(finished, totalToDo) + ' of ' + totalToDo + '\u2026');
            }
        }

        const workers = [];
        const workerCount = Math.min(CONCURRENCY, pending.length);
        for (let i = 0; i < workerCount; i++) workers.push(worker());
        await Promise.all(workers);

        bulkRunning = false;
        updateMainButton();
        refreshChrome();

        const total = state.items.length;
        const converted = state.items.filter((i) => i.status === 'converted').length;
        if (!total) {
            setStatusText('');
        } else if (converted === total) {
            setStatusText('All ' + total + ' converted.');
        } else {
            setStatusText(converted + ' of ' + total + ' converted \u00b7 ' + (total - converted) + ' pending.');
        }
    }

    // ========================= ZIP =========================

    async function downloadZip(items, zipName) {
        if (isBusy()) return;

        const converted = items.filter((i) => i.status === 'converted' && i.blob);
        if (!converted.length) return;

        if (typeof JSZip === 'undefined') {
            showToast('Zip is unavailable \u2014 download images one by one.');
            return;
        }

        zipping = true;
        updateMainButton();
        updateSelectionUI();
        setStatusText('Preparing ZIP\u2026');

        try {
            const zip = new JSZip();
            const used = new Set();
            converted.forEach((item) => {
                const target = FORMATS[item.outFormat] || FORMATS[state.to];
                const fallback = stemOf(item.file.name) + '.' + (target ? target.ext : 'bin');
                zip.file(uniqueName(item.filename || fallback, used), item.blob);
            });
            const blob = await zip.generateAsync({ type: 'blob' });
            downloadBlob(blob, zipName);
            setStatusText(converted.length + (converted.length > 1 ? ' files' : ' file') + ' zipped.');
        } catch (err) {
            showToast('Could not build the ZIP file.');
        } finally {
            zipping = false;
            updateMainButton();
            updateSelectionUI();
        }
    }

    // ========================= SELECTION + CHROME =========================

    function setStatusText(text) {
        const el = $('#convert-actionbar-status');
        if (el) el.textContent = text;
    }

    function updateSelectionUI() {
        const selectedItems = state.items.filter((i) => i.selected);
        const selected = selectedItems.length;
        const ready = selectedItems.filter((i) => i.status === 'converted' && i.blob).length;

        const info = $('#convert-select-info');
        if (info) {
            if (!state.items.length) {
                info.textContent = '\u2014';
            } else if (selected === 0) {
                info.textContent = state.items.length + (state.items.length === 1 ? ' file' : ' files');
            } else {
                info.textContent = 'Selected ' + selected + ' of ' + state.items.length + ' files';
            }
        }

        const dlBtn = $('#convert-download-selected-btn');
        if (dlBtn) {
            if (ready > 0 && !isBusy()) {
                dlBtn.classList.remove('hidden');
                dlBtn.disabled = false;
                dlBtn.textContent = 'Download Selected (' + ready + ')';
            } else {
                dlBtn.classList.add('hidden');
                dlBtn.disabled = true;
                dlBtn.textContent = 'Download Selected';
            }
        }
    }

    function updateMainButton() {
        const btn = $('#convert-all-btn');
        if (!btn) return;
        const total = state.items.length;
        const converted = state.items.filter((i) => i.status === 'converted').length;
        const remaining = total - converted;
        const allConverted = total > 0 && converted === total;
        btn.textContent = allConverted
            ? 'Download All (.ZIP)'
            : 'Convert All (' + remaining + ' remaining)';
        btn.disabled = isBusy() || total === 0;
        btn.classList.toggle('is-download', allConverted);
    }

    function updateHeadStats() {
        const target = $('#convert-head-target');
        if (target) target.textContent = targetLabel();
        const quality = $('#convert-head-quality');
        if (quality) {
            const fmt = FORMATS[state.to];
            quality.textContent = (fmt && fmt.lossy) ? (state.quality + '% (Lossy HQ)') : 'Lossless';
        }
    }

    function refreshChrome() {
        const total = state.items.length;
        const converted = state.items.filter((i) => i.status === 'converted').length;
        const selected = state.items.filter((i) => i.selected).length;
        const hasItems = total > 0;

        const listSection = $('#convert-list-section');
        const actionbar = $('#convert-actionbar');
        if (listSection) listSection.classList.toggle('hidden', !hasItems);
        if (actionbar) actionbar.classList.toggle('hidden', !hasItems);

        const countEl = $('#convert-actionbar-count');
        if (countEl) countEl.textContent = selected + (selected === 1 ? ' file selected' : ' files selected');

        updateHeadStats();

        const selectAll = $('#convert-select-all');
        if (selectAll) {
            selectAll.checked = hasItems && selected === total;
            selectAll.indeterminate = selected > 0 && selected < total;
            selectAll.disabled = !hasItems;
        }

        updateSelectionUI();
        updateMainButton();

        if (!isBusy()) {
            if (!total) {
                setStatusText('');
            } else if (converted === total) {
                setStatusText(total + (total === 1 ? ' image converted.' : ' images converted.'));
            } else if (converted === 0) {
                setStatusText(total + (total === 1 ? ' image ready.' : ' images ready.'));
            } else {
                setStatusText(converted + ' of ' + total + ' converted.');
            }
        }
    }

    function setAllSelected(checked) {
        state.items.forEach((item) => {
            item.selected = checked;
            if (item.el && item.el.checkbox) item.el.checkbox.checked = checked;
            if (item.el && item.el.root) item.el.root.classList.toggle('is-selected', checked);
        });
        refreshChrome();
    }

    function selectConverted() {
        state.items.forEach((item) => {
            const on = item.status === 'converted';
            item.selected = on;
            if (item.el && item.el.checkbox) item.el.checkbox.checked = on;
            if (item.el && item.el.root) item.el.root.classList.toggle('is-selected', on);
        });
        refreshChrome();
    }

    // Invalidates any converted item whose stored output format no longer
    // matches the currently selected target; also refreshes every "Output:"
    // label. Never lets a stale blob be downloaded.
    function applyTarget() {
        const target = FORMATS[state.to];
        const label = target ? target.label : String(state.to).toUpperCase();

        state.items.forEach((item) => {
            if (item.el && item.el.metaConv) item.el.metaConv.textContent = convLabel(item);
            if (item.status === 'queued' && item.el && item.el.metaResult && !item.el.metaResult.classList.contains('is-converting')) {
                item.el.metaResult.textContent = 'Est. ' + estimateSize(item);
            }
            if (item.status === 'converted' && item.outFormat !== state.to) {
                item.token += 1;
                stopEstimate(item);
                item.blob = null;
                item.filename = null;
                item.outFormat = null;
                setProgress(item, 0);
                setItemStatus(item, 'queued');
            }
        });

        updateTargetBadge();
        refreshChrome();
    }

    function onMainButton() {
        if (isBusy()) return;
        const total = state.items.length;
        if (!total) return;
        const converted = state.items.filter((i) => i.status === 'converted' && i.blob).length;
        if (converted === total) {
            downloadZip(state.items, ZIP_ALL_NAME);
        } else {
            convertAll();
        }
    }

    // ========================= EVENTS =========================

    function bindList() {
        if (!state.list) return;

        state.list.addEventListener('click', (e) => {
            const card = e.target.closest('.convert-item');
            if (!card) return;
            const id = card.dataset.id;

            if (e.target.closest('.convert-remove-btn')) {
                removeItem(id);
                return;
            }

            const btn = e.target.closest('.convert-btn');
            if (!btn) return;
            const item = findItem(id);
            if (!item) return;

            if (btn.classList.contains('convert-btn-download')) {
                downloadItem(item);
            } else {
                convertOne(item);
            }
        });

        state.list.addEventListener('change', (e) => {
            const checkbox = e.target.closest('.convert-item-checkbox');
            if (!checkbox) return;
            const card = checkbox.closest('.convert-item');
            if (!card) return;
            const item = findItem(card.dataset.id);
            if (!item) return;
            item.selected = checkbox.checked;
            card.classList.toggle('is-selected', checkbox.checked);
            updateSelectionUI();
            const selectAll = $('#convert-select-all');
            const total = state.items.length;
            const selected = state.items.filter((i) => i.selected).length;
            if (selectAll) {
                selectAll.checked = total > 0 && selected === total;
                selectAll.indeterminate = selected > 0 && selected < total;
            }
        });
    }

    function init() {
        const fromSel = $('#convert-from');
        const toSel = $('#convert-to');
        if (!fromSel || !toSel) return; // converter tab not present

        state.from = fromSel.value || 'auto';
        state.to = toSel.value || 'jpg';
        state.list = $('#convert-list');

        fromSel.addEventListener('change', () => {
            state.from = fromSel.value;
            updateTargetBadge();
        });

        toSel.addEventListener('change', () => {
            state.to = toSel.value;
            updateQualityVisibility();
            applyTarget();
        });

        const quality = $('#convert-quality');
        const qualityVal = $('#convert-quality-val');
        if (quality) {
            state.quality = parseInt(quality.value, 10) || 90;
            quality.addEventListener('input', () => {
                state.quality = parseInt(quality.value, 10) || 90;
                if (qualityVal) qualityVal.textContent = quality.value;
                updateHeadStats();
            });
            if (qualityVal) qualityVal.textContent = quality.value;
        }

        const okBtn = $('#convert-ok-btn');
        if (okBtn) {
            okBtn.addEventListener('click', () => {
                showUpload();
                applyTarget();
            });
        }

        const backBtn = $('#convert-back-btn');
        if (backBtn) backBtn.addEventListener('click', showSetup);

        bindDropzone('#convert-dropzone', '#convert-file-input', addFiles);
        bindList();

        const selectAll = $('#convert-select-all');
        if (selectAll) selectAll.addEventListener('change', () => setAllSelected(selectAll.checked));

        const quickAll = $('#convert-quick-all');
        if (quickAll) quickAll.addEventListener('click', () => setAllSelected(true));

        const quickNone = $('#convert-quick-none');
        if (quickNone) quickNone.addEventListener('click', () => setAllSelected(false));

        const quickConverted = $('#convert-quick-converted');
        if (quickConverted) quickConverted.addEventListener('click', selectConverted);

        const clearBtn = $('#convert-clear-btn');
        if (clearBtn) clearBtn.addEventListener('click', clearAll);

        const dlSelected = $('#convert-download-selected-btn');
        if (dlSelected) {
            dlSelected.addEventListener('click', () => {
                downloadZip(state.items.filter((i) => i.selected), ZIP_SELECTED_NAME);
            });
        }

        const allBtn = $('#convert-all-btn');
        if (allBtn) allBtn.addEventListener('click', onMainButton);

        updateQualityVisibility();
        updateTargetBadge();
        showSetup();
        refreshChrome();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
