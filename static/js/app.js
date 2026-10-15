/**
 * Upscale4K — Main application controller.
 */
(function () {
    'use strict';

    // --- State ---
    const state = {
        taskId: null,
        videoInfo: null,
        selectedMode: 'basic',
        selectedScale: 4,
        processing: false,
        file: null,
        cancelRequested: false,
        activeUpscaler: null,
    };

    // --- DOM refs ---
    const $ = (sel) => document.querySelector(sel);
    const dropzone = $('#dropzone');
    const fileInput = $('#file-input');
    const uploadSection = $('#upload-section');
    const appSection = $('#app-section');
    const originalVideo = $('#original-video');
    const upscaledVideo = $('#upscaled-video');
    const outputPlaceholder = $('#output-placeholder');
    const upscaleBtn = $('#upscale-btn');
    const progressSection = $('#progress-section');
    const progressFill = $('#progress-fill');
    const progressMessage = $('#progress-message');
    const progressPercent = $('#progress-percent');
    const progressFrames = $('#progress-frames');
    const progressEta = $('#progress-eta');
    const downloadSection = $('#download-section');
    const downloadBtn = $('#download-btn');
    const downloadDetails = $('#download-details');
    const newVideoRow = $('#new-video-row');
    const newVideoBtn = $('#new-video-btn');
    const gpuBadge = $('#gpu-badge');
    const gpuStatus = $('#gpu-status');
    const proWarning = $('#pro-warning');

    // --- Init ---
    checkWebGPU();
    bindEvents();
    initSidebar();
    initAbout();

    async function checkWebGPU() {
        const dot = gpuBadge.querySelector('.badge-dot');
        let webgpu = false;
        if (navigator.gpu) {
            try {
                webgpu = !!(await navigator.gpu.requestAdapter());
            } catch { /* fall through */ }
        }

        if (webgpu) {
            dot.classList.add('active');
            gpuStatus.textContent = 'WebGPU ready';
            return;
        }

        // No WebGPU (e.g. the desktop webview or an older browser). If the
        // server has its own AI engine that's the better path anyway — show
        // a healthy badge and default images to Ultra mode.
        try {
            const caps = await (await fetch('/api/capabilities')).json();
            if (caps.server_ai) {
                dot.classList.add('active');
                gpuStatus.textContent = 'AI engine ready (server)';
                document.querySelector('[data-imgmode="ultra"]')?.click();
                showProWarning('No WebGPU — Pro runs on slower WASM. Basic mode recommended.');
                return;
            }
        } catch { /* server unreachable — fall through */ }

        dot.classList.add('unavailable');
        gpuStatus.textContent = 'WebGPU unavailable';
        showProWarning('WebGPU not supported in this browser. Use Chrome 113+.');
    }

    // Session token for file-writing endpoints (embedded in the page, so
    // only same-origin scripts can read it; the custom header forces a
    // CORS preflight that blocks cross-origin requests).
    const NGU_TOKEN = document.querySelector('meta[name="ngu-token"]')?.content || '';

    // External links: normal browsers open a tab; the desktop webview
    // blocks window.open, so fall back to asking the local server to
    // open the system browser.
    function openExternal(url) {
        const w = window.open(url, '_blank', 'noopener');
        if (!w) {
            fetch('/api/open', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-NGU-Token': NGU_TOKEN },
                body: JSON.stringify({ url }),
            }).catch(() => {});
        }
    }

    // --- About dialog ---
    function initAbout() {
        const overlay = $('#about-overlay');
        $('#about-btn').addEventListener('click', () => overlay.classList.remove('hidden'));
        $('#about-close').addEventListener('click', () => overlay.classList.add('hidden'));
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.classList.add('hidden');
        });

        document.querySelectorAll('a.ext-link').forEach((a) => {
            a.addEventListener('click', (e) => {
                e.preventDefault();
                openExternal(a.href);
            });
        });

        // Fill version everywhere it's shown; warn early if FFmpeg is broken
        fetch('/api/capabilities')
            .then((r) => r.json())
            .then((caps) => {
                if (caps.ffmpeg === false) {
                    showToast('FFmpeg is not available — video upscaling and Quality mode will not work.');
                }
                if (!caps.version) return;
                $('#about-version').textContent = caps.version;
                const sv = $('#sidebar-version');
                if (sv) sv.textContent = 'v' + caps.version;
            })
            .catch(() => {});
    }

    function showProWarning(msg) {
        proWarning.textContent = msg;
        proWarning.classList.remove('hidden');
    }

    function bindEvents() {
        // Dropzone
        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) handleVideoFiles(e.dataTransfer.files);
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) handleVideoFiles(fileInput.files);
        });

        // Mode selection
        document.querySelectorAll('.mode-card').forEach((card) => {
            card.addEventListener('click', () => {
                if (state.processing || card.classList.contains('disabled')) return;
                document.querySelectorAll('.mode-card').forEach((c) => c.classList.remove('selected'));
                card.classList.add('selected');
                state.selectedMode = card.dataset.mode;
                updateOutputRes();
            });
        });

        // Scale buttons
        document.querySelectorAll('.scale-btn').forEach((btn) => {
            btn.addEventListener('click', () => {
                if (state.processing) return;
                document.querySelectorAll('.scale-btn').forEach((b) => b.classList.remove('selected'));
                btn.classList.add('selected');
                state.selectedScale = parseInt(btn.dataset.scale);
                updateOutputRes();
            });
        });

        // Upscale button
        upscaleBtn.addEventListener('click', startUpscale);

        // Cancel button
        $('#cancel-btn').addEventListener('click', cancelVideoJob);

        // Download
        downloadBtn.addEventListener('click', downloadResult);

        // New video
        newVideoBtn.addEventListener('click', resetApp);

        // Video batch
        $('#vbatch-start-btn').addEventListener('click', startVideoBatch);
        $('#vbatch-zip-btn').addEventListener('click', downloadVideoBatchZip);
        $('#vbatch-reset-btn').addEventListener('click', resetVideoBatch);
        document.querySelectorAll('[data-vbatchscale]').forEach((btn) => {
            btn.addEventListener('click', () => {
                if (vbatch.running) return;
                document.querySelectorAll('[data-vbatchscale]').forEach((b) => b.classList.remove('selected'));
                btn.classList.add('selected');
                vbatch.scale = parseInt(btn.dataset.vbatchscale);
            });
        });
    }

    function isValidVideo(file) {
        return (file.type.startsWith('video/') || file.name.match(/\.(mp4|mov|webm|mkv|avi)$/i))
            && file.size <= 2 * 1024 * 1024 * 1024;
    }

    function handleVideoFiles(fileList) {
        const files = Array.from(fileList).filter(isValidVideo);
        if (!files.length) {
            showToast('Please select video files under 2GB.');
            return;
        }
        if (files.length === 1) handleFile(files[0]);
        else enterVideoBatch(files);
    }

    async function handleFile(file) {
        state.file = file;

        // Show preview immediately
        const url = URL.createObjectURL(file);
        originalVideo.src = url;
        originalVideo.load();

        uploadSection.classList.add('hidden');
        appSection.classList.remove('hidden');

        // Upload to server
        upscaleBtn.disabled = true;
        upscaleBtn.innerHTML = spinnerSVG() + ' Uploading...';

        try {
            const formData = new FormData();
            formData.append('video', file);

            const res = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await res.json();

            if (!res.ok) throw new Error(data.error || 'Upload failed');

            state.taskId = data.task_id;
            state.videoInfo = data.info;
            populateInfo(data.info);
            updateOutputRes();

            upscaleBtn.disabled = false;
            upscaleBtn.innerHTML = upscaleIconSVG() + ' Start Upscaling';
            newVideoRow.classList.remove('hidden');
        } catch (err) {
            // Don't strand the user on a broken state — back to the dropzone
            showToast('Upload failed: ' + err.message);
            resetApp();
        }
    }

    function populateInfo(info) {
        $('#info-resolution').textContent = `${info.width} x ${info.height}`;
        $('#info-fps').textContent = info.fps;
        $('#info-duration').textContent = formatDuration(info.duration);
        $('#info-size').textContent = formatSize(info.size);
        $('#info-codec').textContent = info.codec.toUpperCase();
        $('#original-res').textContent = `${info.width}x${info.height}`;
    }

    function updateOutputRes() {
        if (!state.videoInfo) return;
        let w = state.videoInfo.width * state.selectedScale;
        let h = state.videoInfo.height * state.selectedScale;
        // Cap the longer dimension at 3840 (4K)
        const maxDim = 3840;
        if (w > maxDim || h > maxDim) {
            if (w >= h) { const r = maxDim / w; w = maxDim; h = Math.round(h * r); }
            else { const r = maxDim / h; h = maxDim; w = Math.round(w * r); }
        }
        h = h - (h % 2);
        w = w - (w % 2);
        $('#output-resolution').textContent = `${w} x ${h}`;
    }

    function cancelVideoJob() {
        if (!state.processing) return;
        state.cancelRequested = true;
        if (state.activeUpscaler) state.activeUpscaler.cancel();
        if (state.taskId) {
            fetch(`/api/cancel/${state.taskId}`, { method: 'POST' }).catch(() => {});
        }
    }

    function onUpscaleCancelled() {
        state.processing = false;
        state.activeUpscaler = null;
        progressSection.classList.add('hidden');
        newVideoRow.classList.remove('hidden');
        upscaleBtn.disabled = false;
        upscaleBtn.innerHTML = upscaleIconSVG() + ' Start Upscaling';
        showToast('Cancelled — pick any mode and start again.');
    }

    async function startUpscale() {
        if (state.processing || !state.taskId) return;
        state.processing = true;
        state.cancelRequested = false;

        upscaleBtn.disabled = true;
        progressSection.classList.remove('hidden');
        downloadSection.classList.add('hidden');
        newVideoRow.classList.add('hidden');

        const startTime = Date.now();

        if (state.selectedMode === 'basic') {
            await runBasicUpscale(startTime);
        } else {
            await runProUpscale(startTime);
        }
    }

    // --- Basic Mode (server-side FFmpeg) ---
    async function runBasicUpscale(startTime) {
        try {
            const res = await fetch('/api/upscale/basic', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: state.taskId, scale: state.selectedScale }),
            });
            if (!res.ok) throw new Error('Failed to start upscale');

            // Listen to SSE progress
            const evtSource = new EventSource(`/api/progress/${state.taskId}`);
            evtSource.onmessage = (e) => {
                const data = JSON.parse(e.data);
                updateProgress(data, startTime);

                if (data.status === 'completed') {
                    evtSource.close();
                    onUpscaleComplete();
                } else if (data.status === 'cancelled') {
                    evtSource.close();
                    onUpscaleCancelled();
                } else if (data.status === 'error') {
                    evtSource.close();
                    onUpscaleError(data.message);
                }
            };
            evtSource.onerror = () => {
                // The event stream can drop during long encodes while FFmpeg
                // keeps working — switch to polling instead of failing.
                evtSource.close();
                pollVideoProgress(startTime);
            };
        } catch (err) {
            onUpscaleError(err.message);
        }
    }

    async function fetchProgressOnce(taskId) {
        const res = await fetch(`/api/progress/${taskId}`);
        const reader = res.body.getReader();
        const { value } = await reader.read();
        reader.cancel();
        const text = new TextDecoder().decode(value);
        const match = text.match(/data: (.+)/);
        return match ? JSON.parse(match[1]) : null;
    }

    function pollVideoProgress(startTime) {
        let failures = 0;
        const iv = setInterval(async () => {
            try {
                const data = await fetchProgressOnce(state.taskId);
                if (!data) return;
                failures = 0;
                updateProgress(data, startTime);
                if (data.status === 'completed') { clearInterval(iv); onUpscaleComplete(); }
                else if (data.status === 'cancelled') { clearInterval(iv); onUpscaleCancelled(); }
                else if (data.status === 'error') { clearInterval(iv); onUpscaleError(data.message); }
            } catch {
                failures++;
                if (failures > 15) { clearInterval(iv); onUpscaleError('Connection lost'); }
            }
        }, 1000);
    }

    // --- Pro Mode (browser WebGPU) ---

    /**
     * Run one video through the browser AI pipeline: extract frames,
     * upscale each, post them to the server, assemble, wait for completion.
     * Shared by single Pro mode and the batch queue.
     * onProgress(pct, message, currentFrame, totalFrames)
     */
    async function proProcessVideo(file, taskId, info, upscaler, onProgress, isCancelled) {
        await fetch('/api/pro/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskId }),
        });

        const video = document.createElement('video');
        video.muted = true;
        video.playsInline = true;
        video.src = URL.createObjectURL(file);
        await new Promise((resolve) => { video.onloadedmetadata = resolve; video.load(); });

        const fps = info.fps;
        const totalFrames = Math.floor(video.duration * fps);
        const canvas = $('#frame-canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        let processed = 0;

        // Show tile progress within each frame
        upscaler.onTileProgress = (done, total) => {
            const overallPct = Math.min(95, Math.round(((processed + done / total) / totalFrames) * 95));
            onProgress(overallPct,
                `AI upscaling frame ${processed + 1}/${totalFrames} (tile ${done}/${total})`,
                processed, totalFrames);
        };

        try {
            // Seek to each frame individually. Slower than
            // requestVideoFrameCallback streaming, but frame-exact: rVFC
            // silently drops frames when the tab is occluded or throttled,
            // which corrupts the output. Extraction isn't the bottleneck —
            // AI inference is — so correctness wins.
            for (let i = 0; i < totalFrames; i++) {
                if (isCancelled()) throw new Error('cancelled');
                video.currentTime = i / fps;
                await new Promise((resolve) => { video.onseeked = resolve; });

                ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
                const imageData = ctx.getImageData(0, 0, video.videoWidth, video.videoHeight);

                // AI upscale (runs in Web Worker — UI stays responsive)
                const upscaled = await upscaler.processFrame(imageData);

                const outCanvas = $('#output-canvas');
                outCanvas.width = upscaled.width;
                outCanvas.height = upscaled.height;
                outCanvas.getContext('2d').putImageData(upscaled, 0, 0);

                const blob = await new Promise((resolve) => outCanvas.toBlob(resolve, 'image/jpeg', 0.95));

                await fetch(`/api/pro/frame/${taskId}/${processed}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'image/jpeg' },
                    body: blob,
                });

                processed++;
                onProgress(Math.min(95, Math.round((processed / totalFrames) * 95)),
                    `AI upscaling frame ${processed}/${totalFrames}`,
                    processed, totalFrames);
            }
        } finally {
            URL.revokeObjectURL(video.src);
        }

        // Assemble final video, then wait for the server to finish
        onProgress(96, 'Assembling video with audio...', totalFrames, totalFrames);
        await fetch('/api/pro/assemble', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskId, fps }),
        });

        while (true) {
            if (isCancelled()) throw new Error('cancelled');
            await new Promise((r) => setTimeout(r, 1000));
            let data;
            try {
                data = await fetchProgressOnce(taskId);
            } catch {
                continue;
            }
            if (!data) continue;
            if (data.status === 'completed') return;
            if (data.status === 'cancelled') throw new Error('cancelled');
            if (data.status === 'error') throw new Error(data.message || 'Assembly failed');
        }
    }

    async function runProUpscale(startTime) {
        if (typeof ProUpscaler === 'undefined') {
            onUpscaleError('Pro upscaler module not loaded.');
            return;
        }

        try {
            const total = state.videoInfo.duration * state.videoInfo.fps;
            updateProgress({ progress: 0, message: 'Loading AI model...', current_frame: 0, total_frames: total }, startTime);

            const upscaler = new ProUpscaler();
            state.activeUpscaler = upscaler;
            await upscaler.init('/static/models/realesr-general-x4v3.onnx');
            const backend = upscaler.getBackend().toUpperCase();
            updateProgress({ progress: 2, message: `Model loaded (${backend})`, current_frame: 0, total_frames: total }, startTime);

            await proProcessVideo(state.file, state.taskId, state.videoInfo, upscaler,
                (pct, msg, cur, tot) => updateProgress({
                    progress: pct, message: `${msg} [${backend}]`,
                    current_frame: cur, total_frames: tot,
                }, startTime),
                () => state.cancelRequested);

            state.activeUpscaler = null;
            upscaler.dispose();
            onUpscaleComplete();
        } catch (err) {
            if (state.cancelRequested || err.message === 'cancelled') {
                onUpscaleCancelled();
                return;
            }
            onUpscaleError('Pro upscale failed: ' + err.message);
        }
    }

    function updateProgress(data, startTime) {
        const pct = data.progress || 0;
        progressFill.style.width = pct + '%';
        progressPercent.textContent = pct + '%';
        progressMessage.textContent = data.message || 'Processing...';
        progressFrames.textContent = `Frame ${data.current_frame || 0}/${data.total_frames || '?'}`;

        if (pct > 5 && startTime) {
            const elapsed = (Date.now() - startTime) / 1000;
            const estimated = (elapsed / pct) * (100 - pct);
            progressEta.textContent = 'ETA: ' + formatDuration(estimated);
        }
    }

    function onUpscaleComplete() {
        state.processing = false;
        progressSection.classList.add('hidden');
        downloadSection.classList.remove('hidden');
        newVideoRow.classList.remove('hidden');
        upscaleBtn.disabled = false;
        upscaleBtn.innerHTML = upscaleIconSVG() + ' Start Upscaling';

        // Show upscaled video preview
        upscaledVideo.src = `/api/download/${state.taskId}`;
        upscaledVideo.classList.remove('hidden');
        outputPlaceholder.classList.add('hidden');
        $('#upscaled-res').textContent = $('#output-resolution').textContent.replace(' x ', 'x');

        downloadDetails.textContent = `${state.selectedMode === 'pro' ? 'AI' : 'FFmpeg'} upscaled to ${$('#output-resolution').textContent}`;
    }

    function onUpscaleError(msg) {
        state.processing = false;
        progressSection.classList.add('hidden');
        newVideoRow.classList.remove('hidden');
        upscaleBtn.disabled = false;
        upscaleBtn.innerHTML = upscaleIconSVG() + ' Start Upscaling';
        showToast('Error: ' + msg);
    }

    // In the desktop app the webview has no download manager, so downloads
    // go through a native Save dialog + the local server. Browsers with the
    // File System Access API (Chrome/Edge) get a real "Save As" dialog too;
    // others fall back to a normal download.
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

    async function saveServerResult(url, taskId, suggestedName) {
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

    function downloadResult() {
        if (!state.taskId) return;
        const stem = state.file ? state.file.name.replace(/\.[^.]+$/, '') : 'video';
        saveServerResult(`/api/download/${state.taskId}`, state.taskId, `${stem}_4K.mp4`);
    }

    function resetApp() {
        state.taskId = null;
        state.videoInfo = null;
        state.processing = false;
        state.file = null;

        uploadSection.classList.remove('hidden');
        appSection.classList.add('hidden');
        progressSection.classList.add('hidden');
        downloadSection.classList.add('hidden');
        newVideoRow.classList.add('hidden');

        originalVideo.src = '';
        upscaledVideo.src = '';
        upscaledVideo.classList.add('hidden');
        outputPlaceholder.classList.remove('hidden');
        fileInput.value = '';
    }

    // ======================= VIDEO BATCH =======================
    // Multiple videos run sequentially through Basic mode (server FFmpeg) —
    // video jobs saturate the CPU, so parallel would only slow everything.

    const vbatch = {
        items: [],       // { file, status, taskId, el }
        running: false,
        scale: 4,
        cancelRequested: false,
        currentTaskId: null,
        activeUpscaler: null,
    };

    function enterVideoBatch(files) {
        vbatch.items = files.map((file) => ({ file, status: 'pending', taskId: null, el: null }));
        vbatch.running = false;
        vbatch.cancelRequested = false;

        uploadSection.classList.add('hidden');
        appSection.classList.add('hidden');
        $('#video-batch-section').classList.remove('hidden');

        $('#vbatch-title').textContent = `Batch: ${vbatch.items.length} videos`;
        $('#vbatch-sub').textContent = 'Pick a mode and scale, then start — videos run one at a time';
        $('#vbatch-progress-label').textContent = '—';
        $('#vbatch-zip-btn').classList.add('hidden');
        $('#vbatch-start-btn').disabled = false;
        $('#vbatch-reset-btn').textContent = 'Cancel';

        const list = $('#vbatch-list');
        list.innerHTML = '';
        vbatch.items.forEach((item) => {
            const row = document.createElement('div');
            row.className = 'vbatch-row';
            const name = document.createElement('span');
            name.className = 'vbatch-name';
            name.textContent = item.file.name;
            const size = document.createElement('span');
            size.className = 'vbatch-size';
            size.textContent = formatSize(item.file.size);
            const bar = document.createElement('div');
            bar.className = 'vbatch-bar';
            const fill = document.createElement('div');
            fill.className = 'vbatch-fill';
            bar.appendChild(fill);
            const status = document.createElement('span');
            status.className = 'batch-status pending';
            status.textContent = 'queued';
            row.append(name, size, bar, status);
            list.appendChild(row);
            item.el = { row, status, fill };
        });
    }

    function setVItem(item, cls, text, pct) {
        item.el.status.className = 'batch-status ' + cls;
        item.el.status.textContent = text;
        if (pct !== undefined) item.el.fill.style.width = pct + '%';
    }

    async function startVideoBatch() {
        if (vbatch.running || !vbatch.items.length) return;
        const mode = $('#vbatch-mode').value;
        if (mode === 'pro' && typeof ProUpscaler === 'undefined') {
            showToast('Pro upscaler module not loaded.');
            return;
        }
        vbatch.running = true;
        vbatch.cancelRequested = false;
        $('#vbatch-start-btn').disabled = true;
        $('#vbatch-zip-btn').classList.add('hidden');
        $('#vbatch-reset-btn').textContent = 'Cancel';

        // Pro mode: load the browser AI model once for the whole batch
        let upscaler = null;
        if (mode === 'pro') {
            try {
                $('#vbatch-sub').textContent = 'Loading AI model...';
                upscaler = new ProUpscaler();
                vbatch.activeUpscaler = upscaler;
                await upscaler.init('/static/models/realesr-general-x4v3.onnx');
                $('#vbatch-sub').textContent = `AI upscaling in your browser (${upscaler.getBackend().toUpperCase()})`;
            } catch (e) {
                showToast('Could not load the AI model: ' + e.message);
                vbatch.running = false;
                vbatch.activeUpscaler = null;
                $('#vbatch-start-btn').disabled = false;
                return;
            }
        } else {
            $('#vbatch-sub').textContent = 'Upscaling on the server (FFmpeg)';
        }

        let done = 0;
        for (const item of vbatch.items) {
            if (vbatch.cancelRequested) break;
            $('#vbatch-progress-label').textContent = `${done + 1} / ${vbatch.items.length}`;
            try {
                await processOneVideo(item, mode, upscaler);
                setVItem(item, 'done', 'done', 100);
                addVItemSaveLink(item);
            } catch (e) {
                if (vbatch.cancelRequested || (e && e.message === 'cancelled')) {
                    setVItem(item, 'failed', 'cancelled');
                    break;
                }
                console.error('video batch item failed', e);
                setVItem(item, 'failed', 'failed');
            }
            done++;
        }
        if (upscaler && upscaler.worker) upscaler.dispose();
        vbatch.activeUpscaler = null;

        const finished = vbatch.items.filter((i) => i.status === 'completed');
        if (vbatch.cancelRequested) {
            $('#vbatch-progress-label').textContent = 'Cancelled';
            $('#vbatch-sub').textContent = `Cancelled — ${finished.length} finished before stopping`;
        } else {
            $('#vbatch-progress-label').textContent = 'Complete';
            $('#vbatch-sub').textContent = `${finished.length} of ${vbatch.items.length} succeeded`;
        }
        if (finished.length) $('#vbatch-zip-btn').classList.remove('hidden');

        vbatch.running = false;
        vbatch.currentTaskId = null;
        vbatch.cancelRequested = false;
        $('#vbatch-start-btn').disabled = false;
        $('#vbatch-reset-btn').textContent = 'Close';
    }

    async function processOneVideo(item, mode, upscaler) {
        setVItem(item, 'working', 'uploading', 0);
        const formData = new FormData();
        formData.append('video', item.file);
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Upload failed');
        item.taskId = data.task_id;
        vbatch.currentTaskId = data.task_id;
        if (vbatch.cancelRequested) throw new Error('cancelled');

        if (mode === 'pro') {
            await proProcessVideo(item.file, data.task_id, data.info, upscaler,
                (pct) => setVItem(item, 'working', pct + '%', pct),
                () => vbatch.cancelRequested);
        } else {
            const upRes = await fetch('/api/upscale/basic', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: data.task_id, scale: vbatch.scale }),
            });
            if (!upRes.ok) throw new Error('Failed to start upscale');

            // Poll progress (same endpoint the single-video fallback uses)
            let failures = 0;
            while (true) {
                if (vbatch.cancelRequested) throw new Error('cancelled');
                await new Promise((r) => setTimeout(r, 1000));
                let s;
                try {
                    s = await fetchProgressOnce(data.task_id);
                    failures = 0;
                } catch {
                    if (++failures > 15) throw new Error('Connection lost');
                    continue;
                }
                if (!s) continue;
                setVItem(item, 'working', (s.progress || 0) + '%', s.progress || 0);
                if (s.status === 'completed') break;
                if (s.status === 'cancelled') throw new Error('cancelled');
                if (s.status === 'error') throw new Error(s.message || 'Server error');
            }
        }
        item.status = 'completed';
        vbatch.currentTaskId = null;
    }

    function addVItemSaveLink(item) {
        const a = document.createElement('a');
        a.textContent = 'save';
        a.href = '#';
        a.className = 'vbatch-save';
        const stem = item.file.name.replace(/\.[^.]+$/, '');
        a.addEventListener('click', (e) => {
            e.preventDefault();
            saveServerResult(`/api/download/${item.taskId}`, item.taskId, `${stem}_4K.mp4`);
        });
        item.el.row.appendChild(a);
    }

    async function downloadVideoBatchZip() {
        const ids = vbatch.items.filter((i) => i.status === 'completed').map((i) => i.taskId);
        if (!ids.length) return;
        const btn = $('#vbatch-zip-btn');
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
            await saveServerResult(`/api/zip/${data.zip_id}`, data.zip_id, 'upscaled_videos.zip');
        } catch (e) {
            showToast('Zip failed: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Download All (.zip)';
        }
    }

    function resetVideoBatch() {
        if (vbatch.running) {
            // While running, this button acts as Cancel
            vbatch.cancelRequested = true;
            if (vbatch.activeUpscaler) vbatch.activeUpscaler.cancel();
            if (vbatch.currentTaskId) {
                fetch(`/api/cancel/${vbatch.currentTaskId}`, { method: 'POST' }).catch(() => {});
            }
            $('#vbatch-sub').textContent = 'Cancelling...';
            return;
        }
        vbatch.items = [];
        $('#vbatch-list').innerHTML = '';
        $('#video-batch-section').classList.add('hidden');
        uploadSection.classList.remove('hidden');
        fileInput.value = '';
    }

    // --- Utils ---
    function formatDuration(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return m > 0 ? `${m}m ${s}s` : `${s}s`;
    }

    function formatSize(bytes) {
        if (bytes > 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
        if (bytes > 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        return (bytes / 1024).toFixed(0) + ' KB';
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

    function spinnerSVG() {
        return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation:spin 0.8s linear infinite"><circle cx="12" cy="12" r="10" stroke-opacity="0.25"/><path d="M12 2a10 10 0 019.95 9" stroke-opacity="1"/></svg>';
    }

    function upscaleIconSVG() {
        return '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>';
    }

    // --- Sidebar Navigation ---
    function initSidebar() {
        document.querySelectorAll('.nav-item').forEach((item) => {
            item.addEventListener('click', () => {
                document.querySelectorAll('.nav-item').forEach((n) => n.classList.remove('active'));
                item.classList.add('active');
                const tab = item.dataset.tab;
                document.querySelectorAll('.tab-content').forEach((t) => t.classList.remove('active'));
                const target = document.getElementById('tab-' + tab);
                if (target) target.classList.add('active');
            });
        });
    }
})();
