/* Shared progress clock. Renderers own geometry, selection and camera state. */
(function (global) {
  'use strict';
  const timing = Object.freeze({ forward: 1000, reverse: 900, generationHold: 1000, assignmentHold: 1500, manual: 700 });
  const clamp = value => Math.max(0, Math.min(1, value));
  function create({ draw, onChange = () => {}, pauseOnReturn = false, clock, media, visibility } = {}) {
    clock ||= { now: () => global.performance.now(), frame: fn => global.requestAnimationFrame(fn), cancelFrame: id => global.cancelAnimationFrame(id), delay: (fn, ms) => global.setTimeout(fn, ms), cancelDelay: id => global.clearTimeout(id) };
    media ||= global.matchMedia('(prefers-reduced-motion: reduce)');
    visibility ||= global.document;
    let progress = 0, target = 0, active = false, seen = false, held = false, loop = true;
    let frame = null, timer = null, running = false, serial = 0, disposed = false;
    const notify = () => onChange({ progress, target, active, held, loop, playing: running || timer !== null });
    const paint = value => { progress = clamp(value); draw(progress); };
    function clear() {
      serial++;
      if (frame !== null) clock.cancelFrame(frame);
      if (timer !== null) clock.cancelDelay(timer);
      frame = timer = null; running = false;
    }
    function pause() { if (disposed) return; clear(); held = true; notify(); }
    function animateTo(next, manual, done) {
      if (disposed) return;
      clear(); target = next;
      if (media.matches || progress === next || !active) { paint(next); done?.(); notify(); return; }
      running = true;
      const from = progress, start = clock.now(), token = serial;
      const duration = Math.max(100, Math.abs(next - from) * (manual ? timing.manual : next ? timing.forward : timing.reverse));
      notify();
      function tick(now) {
        if (disposed || token !== serial || !active) return;
        const t = clamp((now - start) / duration), eased = t * t * (3 - 2 * t);
        paint(from + (next - from) * eased);
        if (t < 1) frame = clock.frame(tick);
        else { frame = null; running = false; done?.(); notify(); }
      }
      frame = clock.frame(tick);
    }
    function schedule() {
      if (disposed || !active || held || !loop || visibility.hidden) {
        if (!loop) held = true;
        notify(); return;
      }
      const token = serial;
      timer = clock.delay(() => {
        timer = null;
        if (disposed || token !== serial || !active || held || visibility.hidden) return;
        animateTo(progress < .5 ? 1 : 0, false, schedule);
      }, progress < .5 ? timing.generationHold : timing.assignmentHold);
      notify();
    }
    function settle(next = progress < .5 ? 0 : 1) { if (disposed) return; clear(); paint(next); notify(); }
    function setActive(next) {
      if (disposed || active === next) return;
      if (!next) { clear(); active = false; paint(progress < .5 ? 0 : 1); if (seen && pauseOnReturn) held = true; notify(); return; }
      active = true; seen = true; paint(progress < .5 ? 0 : 1);
      if (!held) schedule(); else notify();
    }
    function play() {
      if (disposed || !active) return;
      held = false;
      animateTo(progress > 0 && progress < 1 ? target : progress === 0 ? 1 : 0, false, schedule);
    }
    function goTo(next) { if (disposed) return; pause(); animateTo(next, true); }
    function replay() { if (disposed || !active) return; held = false; animateTo(0, true, schedule); }
    function setLoop(next) {
      loop = next;
      if (!loop) { if (timer !== null) clock.cancelDelay(timer); timer = null; if (!running) held = true; }
      else if (active && !held && !running && timer === null) schedule();
      notify();
    }
    const onVisibility = () => { if (active && visibility.hidden) pause(); };
    const onMedia = () => { if (media.matches) { pause(); settle(); } };
    visibility.addEventListener('visibilitychange', onVisibility);
    media.addEventListener('change', onMedia);
    paint(0); notify();
    return { setActive, play, pause, goTo, replay, setLoop, settle,
      toggle: () => running || timer !== null ? pause() : play(),
      get progress() { return progress; }, get playing() { return running || timer !== null; },
      destroy() { if (disposed) return; clear(); disposed = true; visibility.removeEventListener('visibilitychange', onVisibility); media.removeEventListener('change', onMedia); },
    };
  }
  global.PtoVisualization2MorphPlayer = Object.freeze({ create, timing });
})(window);
