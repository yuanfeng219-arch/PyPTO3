(function attachPtoHardwareArchitectureViewport(global) {
  'use strict';

  const DEFAULT_ZOOM_LEVELS = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2];

  const qs = (selector, root = document) => (
    !selector ? null : selector instanceof Element ? selector : root.querySelector(selector)
  );
  const qsa = (selector, root = document) => (
    !selector ? [] : Array.from(root.querySelectorAll(selector))
  );
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  function closestZoomLevel(levels, value) {
    const normalized = Array.isArray(levels) && levels.length ? levels : DEFAULT_ZOOM_LEVELS;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return normalized.includes(1) ? 1 : normalized[0];
    return normalized.reduce((closest, level) => (
      Math.abs(level - numeric) < Math.abs(closest - numeric) ? level : closest
    ), normalized[0]);
  }

  function fitZoomLevel(levels, value) {
    const normalized = Array.isArray(levels) && levels.length ? levels : DEFAULT_ZOOM_LEVELS;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return normalized.includes(1) ? 1 : normalized[0];
    let candidate = normalized[0];
    for (const level of normalized) {
      if (level <= numeric) candidate = level;
      else break;
    }
    return candidate;
  }

  function zoomIndex(levels, value) {
    return Math.max(0, levels.indexOf(closestZoomLevel(levels, value)));
  }

  function mount(rootInput, options = {}) {
    const root = qs(rootInput);
    if (!root) return null;

    const levels = options.zoomLevels || DEFAULT_ZOOM_LEVELS;
    const state = {
      frameReady: options.mode === 'inline',
      detailsVisible: options.detailsVisible !== false,
      scale: closestZoomLevel(levels, options.scale || options.defaultScale || 1),
      fitted: options.fitted !== false,
      frameSize: options.frameSize || { width: 1200, height: 820 },
      pan: {
        x: Number(options.panX) || 0,
        y: Number(options.panY) || 0,
      },
    };
    let activePan = null;

    const elements = {
      viewport: qs(options.viewport, root),
      scale: qs(options.scaleEl, root),
      frame: qs(options.frame, root),
      archSelect: qs(options.archSelect, root),
      detailToggle: qs(options.detailToggle, root),
      zoomOut: qs(options.zoomOut, root),
      zoomIn: qs(options.zoomIn, root),
      fit: qs(options.fit, root),
      actual: qs(options.actual, root),
      readout: qs(options.readout, root),
      pathButtons: qsa(options.pathButtons, root),
    };

    function syncReadout() {
      if (elements.readout) elements.readout.textContent = `${Math.round(state.scale * 100)}%`;
      const index = zoomIndex(levels, state.scale);
      if (elements.zoomOut) elements.zoomOut.disabled = index <= 0;
      if (elements.zoomIn) elements.zoomIn.disabled = index >= levels.length - 1;
    }

    function applyPan() {
      if (!elements.scale) return;
      const pan = `translate3d(${Math.round(state.pan.x)}px, ${Math.round(state.pan.y)}px, 0)`;
      elements.scale.style.transform = elements.frame || options.scaleVariable
        ? pan
        : `${pan} scale(${state.scale})`;
    }

    function applyScale(scale, updateFitted = false) {
      state.scale = closestZoomLevel(levels, scale);
      if (!updateFitted) state.fitted = false;

      if (options.scaleVariable && elements.viewport) {
        elements.viewport.style.setProperty(options.scaleVariable, String(state.scale));
      } else if (elements.frame && elements.scale) {
        elements.frame.style.width = `${state.frameSize.width}px`;
        elements.frame.style.height = `${state.frameSize.height}px`;
        elements.frame.style.transform = `scale(${state.scale})`;
        elements.scale.style.width = `${Math.ceil(state.frameSize.width * state.scale)}px`;
        elements.scale.style.height = `${Math.ceil(state.frameSize.height * state.scale)}px`;
      }
      applyPan();

      syncReadout();
      postToFrame({ type: 'hardware-scale', scale: state.scale });
      options.onScaleChange?.(state.scale, api);
    }

    function fit() {
      if (!elements.viewport) {
        applyScale(1, true);
        return;
      }
      const fitPaddingX = options.fitPaddingX ?? 28;
      const fitPaddingY = options.fitPaddingY ?? 76;
      const availableWidth = Math.max(1, elements.viewport.clientWidth - fitPaddingX);
      const availableHeight = Math.max(1, elements.viewport.clientHeight - fitPaddingY);
      const naturalWidth = Math.max(1, state.frameSize.width);
      const naturalHeight = Math.max(1, state.frameSize.height);
      const rawScale = Math.min(1, availableWidth / naturalWidth, availableHeight / naturalHeight);
      const nextScale = fitZoomLevel(levels, rawScale);
      state.fitted = true;
      if (options.centerOnFit !== false) {
        const originX = elements.scale?.offsetLeft || 0;
        const originY = elements.scale?.offsetTop || 0;
        state.pan.x = (elements.viewport.clientWidth - naturalWidth * nextScale) / 2 - originX;
        state.pan.y = (elements.viewport.clientHeight - naturalHeight * nextScale) / 2 - originY;
      }
      applyScale(nextScale, true);
    }

    function actual() {
      state.fitted = false;
      applyScale(1);
    }

    function stepZoom(direction) {
      const currentIndex = zoomIndex(levels, state.scale);
      const nextIndex = clamp(currentIndex + direction, 0, levels.length - 1);
      applyScale(levels[nextIndex]);
    }

    function scaleAtPoint(scale, clientX, clientY) {
      if (!elements.scale) {
        applyScale(scale);
        return;
      }
      const currentScale = state.scale;
      const nextScale = closestZoomLevel(levels, scale);
      if (nextScale === currentScale) return;
      const beforeRect = elements.scale.getBoundingClientRect();
      const anchorX = (clientX - beforeRect.left) / currentScale;
      const anchorY = (clientY - beforeRect.top) / currentScale;
      applyScale(nextScale);
      const afterRect = elements.scale.getBoundingClientRect();
      state.pan.x += clientX - (afterRect.left + anchorX * nextScale);
      state.pan.y += clientY - (afterRect.top + anchorY * nextScale);
      applyPan();
      options.onPanChange?.({ ...state.pan }, api);
    }

    function stepZoomAtPoint(direction, clientX, clientY) {
      const currentIndex = zoomIndex(levels, state.scale);
      const nextIndex = clamp(currentIndex + direction, 0, levels.length - 1);
      scaleAtPoint(levels[nextIndex], clientX, clientY);
    }

    function postToFrame(message) {
      if (!elements.frame?.contentWindow || !state.frameReady) return false;
      elements.frame.contentWindow.postMessage(message, '*');
      return true;
    }

    function setDetailsVisible(visible) {
      state.detailsVisible = visible !== false;
      if (elements.detailToggle) {
        elements.detailToggle.textContent = state.detailsVisible ? (options.detailOnText || '细节开') : (options.detailOffText || '细节关');
        elements.detailToggle.title = state.detailsVisible
          ? (options.detailOnTitle || '隐藏细节数据')
          : (options.detailOffTitle || '显示细节数据');
        elements.detailToggle.setAttribute('aria-label', elements.detailToggle.title);
        elements.detailToggle.setAttribute('aria-pressed', state.detailsVisible ? 'true' : 'false');
      }

      if (options.mode === 'inline') {
        const host = qs(options.inlineHost || options.scaleEl || options.viewport, root);
        global.PtoMemoryArchitecturePattern?.setDetailVisibility?.(host, state.detailsVisible);
      } else {
        postToFrame({ type: 'hardware-details', visible: state.detailsVisible });
      }

      options.onDetailChange?.(state.detailsVisible, api);
    }

    function setArch(id) {
      if (!id) return;
      options.onArchChange?.(id, api);
      const preset = options.presetForArch?.(id);
      if (preset) postToFrame({ type: 'hardware-arch-change', preset });
    }

    function setPathKind(kind) {
      elements.pathButtons.forEach((button) => {
        button.classList.toggle('is-selected', button.dataset.pathKind === kind);
      });
      options.onPathKindChange?.(kind, api);
    }

    function setFrameSize(width, height) {
      state.frameSize = {
        width: Math.max(1, Number(width) || state.frameSize.width),
        height: Math.max(1, Number(height) || state.frameSize.height),
      };
      if (state.fitted) fit();
      else applyScale(state.scale, true);
    }

    function markReady() {
      state.frameReady = true;
      postToFrame({ type: 'hardware-scale', scale: state.scale });
      setDetailsVisible(state.detailsVisible);
      if (elements.archSelect) setArch(elements.archSelect.value);
      options.onReady?.(api);
    }

    function finishPan(event) {
      if (!activePan || event.pointerId !== activePan.pointerId) return;
      elements.viewport?.classList.remove('is-panning');
      try {
        elements.viewport?.releasePointerCapture?.(event.pointerId);
      } catch (error) {
        // Pointer capture may already be released by a cancel/lost event.
      }
      activePan = null;
    }

    function handlePointerDown(event) {
      if (options.pan === false) return;
      if (!elements.viewport || event.button !== 0) return;
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest?.('button, input, textarea, select, a')) return;
      activePan = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        originX: state.pan.x,
        originY: state.pan.y,
      };
      elements.viewport.classList.add('is-panning');
      elements.viewport.setPointerCapture?.(event.pointerId);
      event.preventDefault();
    }

    function handlePointerMove(event) {
      if (!activePan || event.pointerId !== activePan.pointerId) return;
      state.pan.x = activePan.originX + event.clientX - activePan.startX;
      state.pan.y = activePan.originY + event.clientY - activePan.startY;
      applyPan();
      options.onPanChange?.({ ...state.pan }, api);
      event.preventDefault();
    }

    function handleWheel(event) {
      if (options.wheelZoom === false || (!event.metaKey && !event.ctrlKey)) return;
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest?.('button, input, textarea, select, a')) return;
      const dominantDelta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
      if (!dominantDelta) return;
      event.preventDefault();
      stepZoomAtPoint(dominantDelta < 0 ? 1 : -1, event.clientX, event.clientY);
    }

    function handleMessage(event) {
      if (!elements.frame || event.source !== elements.frame.contentWindow) return;
      if (!event.data || (event.data.type !== 'hardware-ready' && event.data.type !== 'hardware-size')) return;
      setFrameSize(event.data.width, event.data.height);
      if (event.data.type === 'hardware-ready') markReady();
    }

    elements.detailToggle?.addEventListener('click', () => setDetailsVisible(!state.detailsVisible));
    elements.zoomOut?.addEventListener('click', () => stepZoom(-1));
    elements.zoomIn?.addEventListener('click', () => stepZoom(1));
    elements.fit?.addEventListener('click', fit);
    elements.actual?.addEventListener('click', actual);
    elements.archSelect?.addEventListener('change', () => setArch(elements.archSelect.value));
    elements.pathButtons.forEach((button) => {
      button.addEventListener('click', () => setPathKind(button.dataset.pathKind));
    });
    elements.frame?.addEventListener('load', () => requestAnimationFrame(markReady));
    elements.viewport?.addEventListener('pointerdown', handlePointerDown);
    elements.viewport?.addEventListener('pointermove', handlePointerMove);
    elements.viewport?.addEventListener('pointerup', finishPan);
    elements.viewport?.addEventListener('pointercancel', finishPan);
    elements.viewport?.addEventListener('lostpointercapture', finishPan);
    elements.viewport?.addEventListener('wheel', handleWheel, { passive: false });
    global.addEventListener('message', handleMessage);
    global.addEventListener('resize', () => {
      if (state.fitted) fit();
    });

    const api = {
      root,
      state,
      elements,
      applyScale,
      fit,
      actual,
      stepZoom,
      stepZoomAtPoint,
      scaleAtPoint,
      applyPan,
      setDetailsVisible,
      setArch,
      setPathKind,
      setFrameSize,
      markReady,
      postToFrame,
      destroy() {
        elements.viewport?.removeEventListener('pointerdown', handlePointerDown);
        elements.viewport?.removeEventListener('pointermove', handlePointerMove);
        elements.viewport?.removeEventListener('pointerup', finishPan);
        elements.viewport?.removeEventListener('pointercancel', finishPan);
        elements.viewport?.removeEventListener('lostpointercapture', finishPan);
        elements.viewport?.removeEventListener('wheel', handleWheel);
        global.removeEventListener('message', handleMessage);
      },
    };

    setDetailsVisible(state.detailsVisible);
    applyScale(state.scale, true);
    if (options.fitOnMount) requestAnimationFrame(fit);

    return api;
  }

  global.PtoHardwareArchitectureViewport = {
    DEFAULT_ZOOM_LEVELS,
    closestZoomLevel,
    zoomIndex,
    mount,
  };
})(window);
