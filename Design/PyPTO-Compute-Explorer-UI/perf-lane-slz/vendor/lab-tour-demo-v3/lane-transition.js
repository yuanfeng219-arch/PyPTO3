/* 用同一份 rank0 trace 的任务标签、核和时间匹配两页的图元。
 * file:// 的父子页通过 postMessage 交换坐标，动画始终由加载页绘制。 */
window.LaneTransition = (() => {
  const blockKey = (tag, core, start, end) =>
    [tag, core, Math.round(start * 10), Math.round(end * 10)].join(':');
  const clamp = t => Math.max(0, Math.min(1, t));
  const phase = (t, start, end) => clamp((t - start) / (end - start));
  const ease = t => t * t * (3 - 2 * t);
  const mix = (a, b, t) => a + (b - a) * t;
  const peerOrigin = location.protocol === 'file:' ? 'null' : location.origin;
  const send = (peer, data) => peer.postMessage(data, peerOrigin === 'null' ? '*' : peerOrigin);

  function preview() {
    const onMessage = event => {
      if (event.source !== parent || event.origin !== peerOrigin) return;
      if (event.data?.type === 'lane-transition:move') {
        document.documentElement.classList.add('lane-moving');
        return;
      }
      if (event.data?.type === 'lane-transition:show') {
        document.documentElement.classList.remove('lane-preview', 'lane-moving');
        window.removeEventListener('message', onMessage);
        requestAnimationFrame(() => send(parent, { type: 'lane-transition:shown' }));
        return;
      }
      if (event.data?.type !== 'lane-transition:measure') return;
      const rect = (node, scale) => {
        const r = node.getBoundingClientRect(), style = getComputedStyle(node);
        return {
          x: r.left, y: r.top, w: r.width, h: r.height, color: style.fill,
          alpha: Number(style.opacity), radius: node.rx.baseVal.value * scale
        };
      };
      const lane = document.getElementById('lane'), cp = document.getElementById('cp');
      const wrap = document.getElementById('lane-wrap').getBoundingClientRect();
      const axis = document.querySelector('.lane-axis-sticky').getBoundingClientRect();
      const matrix = lane.getScreenCTM(), pathScale = cp.getScreenCTM().a;
      const blocks = [...lane.querySelectorAll('[data-lane-block]')].map(node =>
        [node.dataset.laneBlock, rect(node, matrix.a)]);
      const paths = [...cp.querySelectorAll('[data-lane-path]')].map(node =>
        rect(node, pathScale));
      send(parent, {
        type: 'lane-transition:geometry', blocks, paths,
        axis: { x: matrix.e + Number(lane.dataset.laneAxisX) * matrix.a,
          pxPerUs: Number(lane.dataset.laneAxisScale) * matrix.a },
        clip: { x: wrap.left, y: axis.bottom, right: wrap.right, bottom: wrap.bottom }
      });
    };
    window.addEventListener('message', onMessage);
  }

  function play({ shell, frame, blocks, clip, axis, prepare, begin, fadeChrome, handoff, fallback }) {
    const canvas = document.createElement('canvas');
    canvas.className = 'lane-transition-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    const width = innerWidth, height = innerHeight, dpr = devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    let animation = 0, finished = false;
    const stop = () => {
      finished = true;
      cancelAnimationFrame(animation);
      clearTimeout(timeout);
      clearTimeout(handoffTimeout);
      window.removeEventListener('message', receive);
      window.removeEventListener('resize', go);
      document.removeEventListener('visibilitychange', onVisibility);
    };
    const go = () => {
      if (finished) return;
      stop();
      canvas.remove();
      fallback();
    };
    const activate = () => {
      if (finished) return;
      stop();
      handoff();
      canvas.remove();
    };
    const onVisibility = () => { if (document.hidden) go(); };
    const timeout = setTimeout(go, 3500);
    let handoffTimeout = 0;
    window.addEventListener('resize', go);
    document.addEventListener('visibilitychange', onVisibility);

    // Convert the destination's computed SVG fills once, including CSS color-mix().
    const swatch = document.createElement('canvas');
    swatch.width = swatch.height = 1;
    const sample = swatch.getContext('2d', { willReadFrequently: true });
    const palette = new Map();
    const rgb = color => {
      if (!palette.has(color)) {
        sample.fillStyle = color;
        sample.fillRect(0, 0, 1, 1);
        palette.set(color, [...sample.getImageData(0, 0, 1, 1).data].slice(0, 3));
      }
      return palette.get(color);
    };
    const pair = (from, to) => ({ from, to, colors: [rgb(from.color), rgb(to.color)] });
    const blends = new Map();
    let blendsAt = -1;
    function fill({ colors, from, to }, tone) {
      if (tone === 0) return from.color;
      if (tone === 1) return to.color;
      if (tone !== blendsAt) { blendsAt = tone; blends.clear(); }
      const key = from.color + '/' + to.color;
      if (!blends.has(key)) {
        blends.set(key, 'rgb(' + colors[0].map((v, i) => Math.round(mix(v, colors[1][i], tone))).join(' ') + ')');
      }
      return blends.get(key);
    }
    const paint = (shape, horizontal, vertical, tone, opacity = tone) => {
      const { from, to } = shape;
      ctx.globalAlpha = mix(from.alpha, to.alpha, opacity);
      if (ctx.globalAlpha < .004) return;
      ctx.fillStyle = fill(shape, tone);
      const x = mix(from.x, to.x, horizontal), y = mix(from.y, to.y, vertical);
      const w = mix(from.w, to.w, horizontal), h = mix(from.h, to.h, vertical);
      const radius = mix(0, to.radius, vertical);
      if (radius > .35) {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, radius);
        ctx.fill();
      } else {
        ctx.fillRect(x, y, w, h);
      }
    };
    function receive(event) {
      if (event.source !== frame.contentWindow || event.origin !== peerOrigin) return;
      if (event.data?.type === 'lane-transition:shown' && handoffTimeout) {
        activate();
        return;
      }
      if (event.data?.type !== 'lane-transition:geometry') return;
      const { blocks: targets, paths, clip: endClip, axis: endAxis } = event.data;
      const bounds = frame.getBoundingClientRect();
      // The result is measured inside the Agent tab. Draw in the outer viewport.
      const inViewport = rect => ({ ...rect, x: rect.x + bounds.left, y: rect.y + bounds.top });
      endAxis.x += bounds.left;
      endClip.x += bounds.left; endClip.right += bounds.left;
      endClip.y += bounds.top; endClip.bottom += bounds.top;
      const scale = endAxis.pxPerUs / axis.pxPerUs;
      if (!Number.isFinite(scale) || scale <= 0) { go(); return; }
      const replacements = [];
      const movers = targets.flatMap(([key, rect]) => {
        const from = blocks.get(key);
        if (!from) return [];
        const to = inViewport(rect);
        const x = endAxis.x + (from.x - axis.x) * scale;
        // The result demo lengthens one block and shifts later tasks. Keep those
        // blocks on the shared time axis while moving, then dissolve into their new spans.
        if (Math.abs(x - to.x) > .75 || Math.abs(from.w * scale - to.w) > 2) {
          replacements.push(pair({ ...to, alpha: 0 }, to));
          return [pair(from, { ...to, x, w: from.w * scale, alpha: 0 })];
        }
        return [pair(from, to)];
      });
      const steps = paths.map(rect => {
        const to = inViewport(rect);
        return pair({ ...to, y: to.y + 10, alpha: 0 }, to);
      });
      if (movers.length !== blocks.size || !steps.length) { go(); return; }
      clearTimeout(timeout);

      begin();
      document.body.append(canvas);
      send(frame.contentWindow, { type: 'lane-transition:move' });
      const start = performance.now();
      const tick = now => {
        const t = now - start;
        const pan = ease(phase(t, 0, 710));
        const rows = ease(phase(t, 165, 960));
        const settle = ease(phase(t, 690, 1040));
        const reveal = ease(phase(t, 930, 1230));
        fadeChrome(1 - ease(phase(t, 0, 375)));
        shell.style.opacity = ease(phase(t, 120, 730));
        ctx.clearRect(0, 0, width, height);
        ctx.save();
        const left = mix(clip.x, Math.max(endClip.x, endAxis.x), pan), top = mix(clip.y, endClip.y, rows);
        const right = mix(clip.right, endClip.right, pan), bottom = mix(clip.bottom, endClip.bottom, rows);
        ctx.beginPath();
        ctx.rect(left, top, Math.max(0, right - left), Math.max(0, bottom - top));
        ctx.clip();
        movers.forEach(block => {
          if (mix(block.from.x + block.from.w, block.to.x + block.to.w, pan) <= left ||
              mix(block.from.x, block.to.x, pan) >= right ||
              mix(block.from.y + block.from.h, block.to.y + block.to.h, rows) <= top ||
              mix(block.from.y, block.to.y, rows) >= bottom) return;
          paint(block, pan, rows, settle, block.to.alpha ? settle : reveal);
        });
        if (reveal) replacements.forEach(block => paint(block, 1, 1, reveal));
        ctx.restore();
        steps.forEach((step, i) => {
          const delay = 965 + i * 5;
          const position = ease(phase(t, delay, delay + 260));
          paint(step, 1, position, position);
        });
        if (t < 1380) {
          animation = requestAnimationFrame(tick);
        } else {
          handoffTimeout = setTimeout(go, 500);
          send(frame.contentWindow, { type: 'lane-transition:show' });
        }
      };
      tick(start);
    }
    window.addEventListener('message', receive);
    frame.addEventListener('load', () => {
      // Measure after the source has finished desaturating.
      const start = performance.now();
      const desaturate = now => {
        const p = phase(now - start, 0, 260);
        prepare(ease(p));
        if (p < 1) animation = requestAnimationFrame(desaturate);
        else send(frame.contentWindow, { type: 'lane-transition:measure' });
      };
      animation = requestAnimationFrame(desaturate);
    }, { once: true });
    document.body.append(shell);
  }

  return { blockKey, preview, play };
})();
