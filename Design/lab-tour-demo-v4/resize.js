  // 拖动源码区边界；同时支持键盘微调。
  const codePane = document.getElementById('hpCodePane');
  const resizeHandle = document.getElementById('hpCodeResize');
  const mainPane = document.querySelector('.hp-main');
  function resizeBounds() {
    const width = mainPane.getBoundingClientRect().width;
    return window.innerWidth > 1280
      ? { min: 340, max: Math.max(340, width - (storyPane.hidden ? 560 : 920)) }
      : { min: 280, max: Math.max(280, width - 360) };
  }
  function setCodeWidth(width) {
    const bounds = resizeBounds();
    const value = Math.round(Math.max(bounds.min, Math.min(bounds.max, width)));
    mainPane.style.setProperty('--code-width', value + 'px');
    resizeHandle.setAttribute('aria-valuemin', bounds.min);
    resizeHandle.setAttribute('aria-valuemax', Math.round(bounds.max));
    resizeHandle.setAttribute('aria-valuenow', value);
  }
  resizeHandle.addEventListener('pointerdown', function (e) {
    if (e.button !== 0) return;
    e.preventDefault();
    const startX = e.clientX, startWidth = codePane.getBoundingClientRect().width;
    resizeHandle.setPointerCapture(e.pointerId);
    resizeHandle.classList.add('is-dragging');
    document.body.classList.add('is-resizing');
    function move(event) { setCodeWidth(startWidth + event.clientX - startX); }
    function end() {
      resizeHandle.classList.remove('is-dragging');
      document.body.classList.remove('is-resizing');
      resizeHandle.removeEventListener('pointermove', move);
      resizeHandle.removeEventListener('pointerup', end);
      resizeHandle.removeEventListener('pointercancel', end);
      resizeHandle.removeEventListener('lostpointercapture', end);
    }
    resizeHandle.addEventListener('pointermove', move);
    resizeHandle.addEventListener('pointerup', end);
    resizeHandle.addEventListener('pointercancel', end);
    resizeHandle.addEventListener('lostpointercapture', end);
  });
  resizeHandle.addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    setCodeWidth(codePane.getBoundingClientRect().width + (e.key === 'ArrowLeft' ? -20 : 20));
  });
  function syncResizeHandle() {
    const bounds = resizeBounds();
    resizeHandle.setAttribute('aria-valuemin', bounds.min);
    resizeHandle.setAttribute('aria-valuemax', Math.round(bounds.max));
    resizeHandle.setAttribute('aria-valuenow', Math.round(codePane.getBoundingClientRect().width));
  }
  window.addEventListener('resize', syncResizeHandle);
  syncResizeHandle();

  // 拖动中栏的横向边界，在两个画布间分配高度。
  const midPane = document.querySelector('.hp-mid');
  const logicPane = document.getElementById('hpLogicPane');
  const verticalHandle = document.getElementById('hpVerticalResize');
  function verticalBounds() {
    return { min: 140, max: Math.max(140, midPane.clientHeight - 240) };
  }
  function setLogicHeight(height) {
    const bounds = verticalBounds();
    const value = Math.round(Math.max(bounds.min, Math.min(bounds.max, height)));
    midPane.style.setProperty('--logic-height', value + 'px');
    verticalHandle.setAttribute('aria-valuemin', bounds.min);
    verticalHandle.setAttribute('aria-valuemax', bounds.max);
    verticalHandle.setAttribute('aria-valuenow', value);
  }
  verticalHandle.addEventListener('pointerdown', function (e) {
    if (e.button !== 0) return;
    e.preventDefault();
    const startY = e.clientY, startHeight = logicPane.getBoundingClientRect().height;
    verticalHandle.setPointerCapture(e.pointerId);
    verticalHandle.classList.add('is-dragging');
    document.body.classList.add('is-resizing-vertical');
    function move(event) { setLogicHeight(startHeight + event.clientY - startY); }
    function end() {
      verticalHandle.classList.remove('is-dragging');
      document.body.classList.remove('is-resizing-vertical');
      verticalHandle.removeEventListener('pointermove', move);
      verticalHandle.removeEventListener('pointerup', end);
      verticalHandle.removeEventListener('pointercancel', end);
      verticalHandle.removeEventListener('lostpointercapture', end);
    }
    verticalHandle.addEventListener('pointermove', move);
    verticalHandle.addEventListener('pointerup', end);
    verticalHandle.addEventListener('pointercancel', end);
    verticalHandle.addEventListener('lostpointercapture', end);
  });
  verticalHandle.addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    setLogicHeight(logicPane.getBoundingClientRect().height + (e.key === 'ArrowUp' ? -20 : 20));
  });
  function syncVerticalHandle() {
    const bounds = verticalBounds();
    verticalHandle.setAttribute('aria-valuemin', bounds.min);
    verticalHandle.setAttribute('aria-valuemax', bounds.max);
    verticalHandle.setAttribute('aria-valuenow', Math.round(logicPane.getBoundingClientRect().height));
  }
  window.addEventListener('resize', syncVerticalHandle);
  syncVerticalHandle();
