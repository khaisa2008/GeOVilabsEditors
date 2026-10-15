/**
 * GeOViLabs — Shared image viewer utilities for all tool pages.
 *
 * 1. upgradeZoom() : adds a zoom (+/-/reset) toolbar + drag-to-pan (hand)
 *    cursor + Ctrl+wheel zoom + double-click zoom toggle to every
 *    `.image-wrapper` (each card that displays an image).
 * 2. initCompare() : generic before/after comparison slider so every
 *    image feature can show "Original vs Result".
 * 3. bindPaste()   : lets users paste (Ctrl/Cmd+V) an image into the
 *    image dropzones, and injects a small "paste" hint chip on the card.
 */
(function () {
    'use strict';

    const $ = (sel) => document.querySelector(sel);

    const viewers = [];

    // ============================ ZOOM + PAN ============================
    function attachViewer(wrapper, img) {
        if (wrapper.dataset.viewer) return;
        wrapper.dataset.viewer = '1';

        const state = { s: 1, tx: 0, ty: 0, panning: false, startX: 0, startY: 0, baseTx: 0, baseTy: 0 };

        const controls = document.createElement('div');
        controls.className = 'viewer-controls';
        controls.innerHTML =
            '<button type="button" data-act="out" title="Zoom out">' +
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/></svg></button>' +
            '<button type="button" data-act="in" title="Zoom in">' +
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg></button>' +
            '<button type="button" data-act="reset" title="Reset">' +
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg></button>';

        const hint = document.createElement('div');
        hint.className = 'viewer-hint';

        wrapper.appendChild(controls);
        wrapper.appendChild(hint);

        function clamp() {
            const w = wrapper.clientWidth;
            const h = wrapper.clientHeight;
            const dw = img.clientWidth || 1;
            const dh = img.clientHeight || 1;
            const sx = (dw * state.s - w) / 2;
            const sy = (dh * state.s - h) / 2;
            state.tx = Math.max(-sx, Math.min(sx, state.tx));
            state.ty = Math.max(-sy, Math.min(sy, state.ty));
        }

        function apply() {
            clamp();
            if (state.s <= 1.001) {
                state.s = 1;
                state.tx = 0;
                state.ty = 0;
                img.style.transform = '';
                img.style.transformOrigin = '';
                img.style.transition = '';
                wrapper.dataset.zoomed = '0';
                hint.classList.remove('show');
            } else {
                img.style.transformOrigin = '0 0';
                img.style.transition = 'none';
                img.style.transform = 'translate(' + state.tx + 'px, ' + state.ty + 'px) scale(' + state.s + ')';
                wrapper.dataset.zoomed = '1';
                hint.textContent = 'Drag to pan \u00b7 double-click or Ctrl+scroll to zoom';
                hint.classList.add('show');
            }
        }

        controls.addEventListener('pointerdown', (e) => e.stopPropagation());
        controls.addEventListener('mousedown', (e) => e.stopPropagation());

        controls.addEventListener('click', (e) => {
            const btn = e.target.closest('button');
            if (!btn) return;
            const act = btn.dataset.act;
            if (act === 'in') state.s = Math.min(8, state.s * 1.35);
            else if (act === 'out') state.s = Math.max(1, state.s / 1.35);
            else { state.s = 1; state.tx = 0; state.ty = 0; }
            apply();
        });

        wrapper.addEventListener('wheel', (e) => {
            const canZoom = state.s > 1 || e.ctrlKey || e.metaKey;
            if (!canZoom) return;
            e.preventDefault();
            const rect = wrapper.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;
            const f = e.deltaY < 0 ? 1.12 : 0.89;
            const ns = Math.max(1, Math.min(8, state.s * f));
            const ratio = ns / state.s;
            state.tx = cx - (cx - state.tx) * ratio;
            state.ty = cy - (cy - state.ty) * ratio;
            state.s = ns;
            apply();
        }, { passive: false });

        wrapper.addEventListener('pointerdown', (e) => {
            if (state.s <= 1) return;
            if (e.target.closest('.viewer-controls')) return;
            state.panning = true;
            state.startX = e.clientX;
            state.startY = e.clientY;
            state.baseTx = state.tx;
            state.baseTy = state.ty;
            wrapper.classList.add('panning');
            wrapper.setPointerCapture(e.pointerId);
        });

        wrapper.addEventListener('pointermove', (e) => {
            if (!state.panning) return;
            state.tx = state.baseTx + (e.clientX - state.startX);
            state.ty = state.baseTy + (e.clientY - state.startY);
            apply();
        });

        const endPan = () => {
            if (!state.panning) return;
            state.panning = false;
            wrapper.classList.remove('panning');
        };
        wrapper.addEventListener('pointerup', endPan);
        wrapper.addEventListener('pointercancel', endPan);

        // While zoomed, don't leak clicks (e.g. the bg "pick color" handler).
        wrapper.addEventListener('click', (e) => {
            if (state.s > 1) e.stopPropagation();
        }, true);

        wrapper.addEventListener('dblclick', (e) => {
            e.preventDefault();
            const rect = wrapper.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;
            if (state.s > 1) {
                state.s = 1;
                state.tx = 0;
                state.ty = 0;
            } else {
                state.s = 2.5;
                state.tx = wrapper.clientWidth / 2 - cx;
                state.ty = wrapper.clientHeight / 2 - cy;
            }
            apply();
        });

        viewers.push({ wrapper, clamp });
    }

    function upgradeZoom(scope) {
        const root = scope || document;
        const wrappers = Array.from(root.querySelectorAll('.image-wrapper'));
        wrappers.forEach((wrapper) => {
            const img = Array.from(wrapper.querySelectorAll('img')).find((el) => !el.closest('.compare-box'));
            if (img) attachViewer(wrapper, img);
        });
    }

    window.addEventListener('resize', () => {
        viewers.forEach((v) => v.clamp());
    });

    // ========================= COMPARE SLIDER =========================
    function initCompare(cfg) {
        const section = typeof cfg.section === 'string' ? $(cfg.section) : cfg.section;
        const box = typeof cfg.box === 'string' ? $(cfg.box) : cfg.box;
        const before = typeof cfg.before === 'string' ? $(cfg.before) : cfg.before;
        const after = typeof cfg.after === 'string' ? $(cfg.after) : cfg.after;
        const beforeWrap = typeof cfg.beforeWrap === 'string' ? $(cfg.beforeWrap) : cfg.beforeWrap;
        const handle = typeof cfg.handle === 'string' ? $(cfg.handle) : cfg.handle;

        let dragging = false;

        function setPos(clientX) {
            const rect = box.getBoundingClientRect();
            let pct = ((clientX - rect.left) / rect.width) * 100;
            pct = Math.max(2, Math.min(98, pct));
            beforeWrap.style.width = pct + '%';
            handle.style.left = pct + '%';
        }

        function sync() {
            if (!section || section.classList.contains('hidden')) return;
            before.style.width = (after.clientWidth || 0) + 'px';
            before.style.height = (after.clientHeight || 0) + 'px';
        }

        box.addEventListener('pointerdown', (e) => {
            dragging = true;
            box.setPointerCapture(e.pointerId);
            setPos(e.clientX);
        });
        box.addEventListener('pointermove', (e) => { if (dragging) setPos(e.clientX); });
        box.addEventListener('pointerup', () => { dragging = false; });
        box.addEventListener('pointercancel', () => { dragging = false; });
        window.addEventListener('resize', sync);

        return {
            show: (beforeUrl, afterUrl) => {
                if (!beforeUrl || !afterUrl) return;
                before.src = beforeUrl;
                after.src = afterUrl;
                section.classList.remove('hidden');
                beforeWrap.style.width = '50%';
                handle.style.left = '50%';
                after.onload = sync;
                requestAnimationFrame(sync);
            },
            reset: () => section.classList.add('hidden'),
            sync: sync,
        };
    }

    // ======================== PASTE SUPPORT ========================
    function bindPaste(isActive, onFileList, zone) {
        try {
            const content = zone ? zone.querySelector('.dropzone-content') : null;
            if (content && !content.querySelector('.dz-chips')) {
                const chips = document.createElement('div');
                chips.className = 'dz-chips';
                chips.innerHTML = '<span class="dz-chip">' +
                    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
                    '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>' +
                    'Ctrl/Cmd + V to paste an image</span>';
                content.appendChild(chips);
            }
        } catch (e) { /* no-op */ }

        document.addEventListener('paste', (e) => {
            if (isActive && !isActive()) return;
            const cd = e.clipboardData;
            if (!cd || !cd.items) return;
            const items = Array.from(cd.items);
            for (const it of items) {
                if (it.kind === 'file' && it.type && it.type.indexOf('image/') === 0) {
                    const file = it.getAsFile();
                    if (file) {
                        e.preventDefault();
                        onFileList([file]);
                        return;
                    }
                }
            }
        });
    }

    // ========================== PUBLIC API ==========================
    window.GeoviImg = {
        upgradeZoom: upgradeZoom,
        initCompare: initCompare,
        bindPaste: bindPaste,
    };

    function boot() {
        upgradeZoom(document);
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
    window.addEventListener('load', boot);
})();