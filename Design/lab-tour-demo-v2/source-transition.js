/* 源码联动滚动：新目标从当前位置接续，用户主动滚动时立即让出控制。 */
(function () {
  'use strict';
  const motions = new WeakMap();
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  function cancel(box) {
    const motion = motions.get(box);
    if (motion) { cancelAnimationFrame(motion.frame); motions.delete(box); }
  }
  window.workbenchCancelSourceScroll = cancel;
  window.workbenchScrollSource = function (box, top) {
    cancel(box);
    const end = Math.max(0, Math.min(top, box.scrollHeight - box.clientHeight));
    if (reduced.matches || Math.abs(end - box.scrollTop) < 1) {
      motions.delete(box);
      box.scrollTop = end;
      return;
    }
    const start = box.scrollTop, started = performance.now();
    const duration = Math.min(550, Math.max(280, Math.abs(end - start) * 0.45));
    const motion = {};
    motions.set(box, motion);
    function step(now) {
      if (motions.get(box) !== motion) return;
      const t = Math.min(1, (now - started) / duration);
      box.scrollTop = start + (end - start) * (1 - Math.pow(1 - t, 3));
      if (t < 1) motion.frame = requestAnimationFrame(step);
      else motions.delete(box);
    }
    motion.frame = requestAnimationFrame(step);
  };
  ['srcFile', 'hpCode'].forEach(function (id) {
    const box = document.getElementById(id);
    function stop() { cancel(box); }
    box.addEventListener('wheel', stop, { passive: true });
    box.addEventListener('touchstart', stop, { passive: true });
    box.addEventListener('pointerdown', stop, { passive: true });
  });
}());
