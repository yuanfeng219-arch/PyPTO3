(function attachPtoTensorVolumeCanvas(global) {
  'use strict';

  const VALID_TONES = new Set(['neutral', 'input', 'output', 'compute', 'reduction', 'fusion']);
  const VALID_STATES = new Set([
    'base',
    'ghost',
    'padding',
    'window',
    'current',
    'skipped',
  ]);

  const DEPTH_X_RATIO = 0.52;
  const DEPTH_Y_RATIO = 0.38;
  const DEFAULT_PADDING = Object.freeze({ top: 44, right: 38, bottom: 42, left: 52 });

  const TONE_TOKENS = Object.freeze({
    input: '--primary',
    output: '--success',
    compute: '--warning',
    reduction: '--warning',
    fusion: '--accent',
  });

  function positiveInteger(value, fallback = 1) {
    const numeric = Number(value);
    return Number.isFinite(numeric) && numeric > 0 ? Math.floor(numeric) : fallback;
  }

  function numberOr(value, fallback) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : fallback;
  }

  function resolvePadding(input) {
    const padding = input && typeof input === 'object' ? input : {};
    return {
      top: Math.max(0, numberOr(padding.top, DEFAULT_PADDING.top)),
      right: Math.max(0, numberOr(padding.right, DEFAULT_PADDING.right)),
      bottom: Math.max(0, numberOr(padding.bottom, DEFAULT_PADDING.bottom)),
      left: Math.max(0, numberOr(padding.left, DEFAULT_PADDING.left)),
    };
  }

  function cssValue(name, fallback) {
    const value = global.getComputedStyle?.(document.documentElement)?.getPropertyValue(name)?.trim();
    return value || fallback;
  }

  function parseCssColor(value) {
    const input = String(value || '').trim();
    const shortHex = /^#([0-9a-f]{3})$/i.exec(input);
    if (shortHex) {
      return shortHex[1].split('').map((part) => parseInt(`${part}${part}`, 16));
    }
    const longHex = /^#([0-9a-f]{6})(?:[0-9a-f]{2})?$/i.exec(input);
    if (longHex) {
      const numeric = parseInt(longHex[1], 16);
      return [(numeric >> 16) & 255, (numeric >> 8) & 255, numeric & 255];
    }
    const functional = /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i.exec(input);
    if (functional) return functional.slice(1, 4).map(Number);
    return null;
  }

  function resolvedTokenRgb(name, visited = new Set()) {
    if (!name || visited.has(name)) return null;
    visited.add(name);
    const value = cssValue(name, '');
    const directColor = parseCssColor(value);
    if (directColor) return directColor;
    const reference = /var\(\s*(--[\w-]+)/.exec(value);
    return reference ? resolvedTokenRgb(reference[1], visited) : null;
  }

  function tokenRgb(name, fallbackName = '--foreground-secondary') {
    return resolvedTokenRgb(name)
      || resolvedTokenRgb(fallbackName)
      || [128, 128, 128];
  }

  function rgbString(rgb, alpha = 1) {
    const values = rgb.map((value) => Math.max(0, Math.min(255, Math.round(value))));
    return alpha >= 1
      ? `rgb(${values[0]} ${values[1]} ${values[2]})`
      : `rgb(${values[0]} ${values[1]} ${values[2]} / ${alpha})`;
  }

  function mixRgb(source, target, sourceWeight) {
    const weight = Math.max(0, Math.min(1, sourceWeight));
    return source.map((value, index) => value * weight + target[index] * (1 - weight));
  }

  function tokenFaces(tokenName, state) {
    const base = tokenRgb(tokenName);
    const background = tokenRgb('--background');
    const foreground = tokenRgb('--foreground');
    const strength = state === 'current'
      ? 1
      : state === 'window'
        ? 0.9
        : 0.72;
    return {
      top: rgbString(mixRgb(base, background, strength)),
      east: rgbString(mixRgb(base, background, strength * 0.76)),
      south: rgbString(mixRgb(base, background, strength * 0.62)),
      edge: rgbString(mixRgb(base, foreground, 0.58)),
      lineWidth: state === 'current' ? 1.5 : state === 'window' ? 1.15 : 1,
      topLineWidth: state === 'current' ? 2 : state === 'window' ? 1.4 : 1,
    };
  }

  function neutralFaces(state) {
    const foreground = tokenRgb('--foreground');
    if (state === 'ghost') {
      return {
        top: rgbString(tokenRgb('--surface-4'), 0.28),
        east: rgbString(tokenRgb('--surface-3'), 0.22),
        south: rgbString(tokenRgb('--surface-2'), 0.18),
        edge: rgbString(foreground, 0.18),
      };
    }
    if (state === 'padding') {
      return {
        top: 'rgba(90, 96, 104, 0.25)',
        east: 'rgba(54, 60, 68, 0.25)',
        south: 'rgba(42, 48, 56, 0.25)',
        edge: 'rgba(190, 200, 212, 0.18)',
      };
    }
    if (state === 'skipped') {
      return {
        top: rgbString(foreground, 0.08),
        east: rgbString(foreground, 0.055),
        south: rgbString(foreground, 0.045),
        edge: rgbString(foreground, 0.18),
      };
    }
    return {
      top: 'rgb(74, 80, 88)',
      east: 'rgb(59, 65, 74)',
      south: 'rgb(48, 54, 64)',
      edge: 'rgba(10, 12, 16, 0.72)',
      lineWidth: 1,
      topLineWidth: 1,
    };
  }

  function normalizeScene(input) {
    const scene = input && typeof input === 'object' ? input : {};
    const extent = scene.extent && typeof scene.extent === 'object' ? scene.extent : {};
    return {
      extent: {
        columns: positiveInteger(extent.columns),
        rows: positiveInteger(extent.rows),
        depth: positiveInteger(extent.depth),
      },
      axes: {
        columns: String(scene.axes?.columns || 'W'),
        rows: String(scene.axes?.rows || 'H'),
        depth: String(scene.axes?.depth || 'C'),
      },
      voxels: Array.isArray(scene.voxels) ? scene.voxels : [],
    };
  }

  function normalizeVoxel(voxel, extent, index) {
    if (!voxel || typeof voxel !== 'object') {
      console.warn(`[PtoTensorVolumeCanvas] Skipped voxel ${index}: expected an object.`);
      return null;
    }
    const column = Number(voxel.column);
    const row = Number(voxel.row);
    const depth = Number(voxel.depth);
    const validCoordinate = [column, row, depth].every(Number.isFinite)
      && column >= 0 && column < extent.columns
      && row >= 0 && row < extent.rows
      && depth >= 0 && depth < extent.depth;
    if (!validCoordinate) {
      console.warn(`[PtoTensorVolumeCanvas] Skipped voxel ${voxel.id || index}: coordinate is outside scene extent.`);
      return null;
    }
    return {
      id: String(voxel.id || `voxel-${column}-${row}-${depth}`),
      column,
      row,
      depth,
      tone: VALID_TONES.has(voxel.tone) ? voxel.tone : 'neutral',
      state: VALID_STATES.has(voxel.state) ? voxel.state : 'base',
      label: voxel.label == null ? '' : String(voxel.label),
    };
  }

  function facesFor(voxel) {
    if (voxel.state === 'current') return tokenFaces('--warning', 'current');
    if (
      voxel.state === 'ghost'
      || voxel.state === 'padding'
      || voxel.state === 'skipped'
    ) {
      return neutralFaces(voxel.state);
    }

    const state = voxel.state === 'window' ? 'window' : 'base';
    let token = voxel.tone === 'neutral' && state !== 'base'
      ? '--primary'
      : TONE_TOKENS[voxel.tone];
    return token ? tokenFaces(token, state) : neutralFaces(state);
  }

  function point(originX, originY, unit, depthX, depthY, column, row, depth) {
    return {
      x: originX + column * unit + depth * depthX,
      y: originY + row * unit - depth * depthY,
    };
  }

  function drawQuad(ctx, points, fill, stroke, lineWidth = 1) {
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let index = 1; index < points.length; index += 1) ctx.lineTo(points[index].x, points[index].y);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }

  function drawVoxel(ctx, layout, voxel, faces) {
    const gap = 0.065;
    const p = (column, row, depth) => point(
      layout.originX,
      layout.originY,
      layout.unit,
      layout.depthX,
      layout.depthY,
      column,
      row,
      depth
    );
    const c0 = voxel.column + gap;
    const c1 = voxel.column + 1 - gap;
    const r0 = voxel.row + gap;
    const r1 = voxel.row + 1 - gap;
    const d0 = voxel.depth + gap;
    const d1 = voxel.depth + 1 - gap;
    const f0 = p(c0, r0, d0);
    const f1 = p(c1, r0, d0);
    const f2 = p(c1, r1, d0);
    const f3 = p(c0, r1, d0);
    const b0 = p(c0, r0, d1);
    const b1 = p(c1, r0, d1);
    const b2 = p(c1, r1, d1);
    drawQuad(ctx, [f0, f1, b1, b0], faces.top, faces.topEdge || faces.edge, faces.topLineWidth || faces.lineWidth || 1);
    drawQuad(ctx, [f1, f2, b2, b1], faces.east, faces.edge, faces.lineWidth || 1);
    drawQuad(ctx, [f0, f1, f2, f3], faces.south, faces.edge, faces.lineWidth || 1);
    return {
      x: (f0.x + f1.x + f2.x + f3.x) / 4,
      y: (f0.y + f1.y + f2.y + f3.y) / 4,
    };
  }

  function computeLayout(width, height, extent, padding) {
    const innerWidth = Math.max(1, width - padding.left - padding.right);
    const innerHeight = Math.max(1, height - padding.top - padding.bottom);
    const widthUnits = extent.columns + extent.depth * DEPTH_X_RATIO;
    const heightUnits = extent.rows + extent.depth * DEPTH_Y_RATIO;
    const unit = Math.max(3, Math.min(34, innerWidth / Math.max(1, widthUnits), innerHeight / Math.max(1, heightUnits)));
    const depthX = unit * DEPTH_X_RATIO;
    const depthY = unit * DEPTH_Y_RATIO;
    const objectWidth = extent.columns * unit + extent.depth * depthX;
    const objectHeight = extent.rows * unit + extent.depth * depthY;
    return {
      unit,
      depthX,
      depthY,
      originX: padding.left + Math.max(0, (innerWidth - objectWidth) / 2),
      originY: padding.top + extent.depth * depthY + Math.max(0, (innerHeight - objectHeight) / 2),
    };
  }

  function drawAxes(ctx, layout, extent, axes) {
    const p = (column, row, depth) => point(
      layout.originX,
      layout.originY,
      layout.unit,
      layout.depthX,
      layout.depthY,
      column,
      row,
      depth
    );
    const frontTopLeft = p(0, 0, 0);
    const frontBottomLeft = p(0, extent.rows, 0);
    const frontBottomRight = p(extent.columns, extent.rows, 0);
    const backTopLeft = p(0, 0, extent.depth);
    ctx.save();
    ctx.fillStyle = cssValue('--foreground', 'CanvasText');
    ctx.font = `700 13px ${cssValue('--font-mono', 'ui-monospace, monospace')}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(axes.columns, (frontBottomRight.x + frontBottomLeft.x) / 2, frontBottomRight.y + 24);
    ctx.textAlign = 'right';
    ctx.fillText(axes.rows, frontBottomLeft.x - 10, (frontTopLeft.y + frontBottomLeft.y) / 2);
    ctx.textAlign = 'center';
    ctx.fillText(axes.depth, (frontTopLeft.x + backTopLeft.x) / 2, (frontTopLeft.y + backTopLeft.y) / 2 - 14);
    ctx.restore();
  }

  function render(canvas, inputScene, inputOptions = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError('PtoTensorVolumeCanvas.render expects an HTMLCanvasElement.');
    }

    let scene = normalizeScene(inputScene);
    let options = { ...inputOptions };
    let observer = null;
    let frame = 0;
    let destroyed = false;

    canvas.classList.add('pto-tensor-volume-canvas');
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', options.ariaLabel || 'Three-dimensional tensor volume');

    function paint() {
      if (destroyed) return;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(1, rect.width || canvas.clientWidth || 1);
      const height = Math.max(1, rect.height || canvas.clientHeight || 1);
      const dpr = Math.max(1, global.devicePixelRatio || 1);
      const backingWidth = Math.max(1, Math.floor(width * dpr));
      const backingHeight = Math.max(1, Math.floor(height * dpr));
      if (canvas.width !== backingWidth) canvas.width = backingWidth;
      if (canvas.height !== backingHeight) canvas.height = backingHeight;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const layout = computeLayout(width, height, scene.extent, resolvePadding(options.padding));
      if (options.showAxes !== false) drawAxes(ctx, layout, scene.extent, scene.axes);

      const voxels = scene.voxels
        .map((voxel, index) => normalizeVoxel(voxel, scene.extent, index))
        .filter(Boolean)
        .sort((a, b) => (b.depth - a.depth) || (b.row - a.row) || (a.column - b.column));

      const fontFamily = cssValue('--font-mono', 'ui-monospace, monospace');
      ctx.font = `700 10px ${fontFamily}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      voxels.forEach((voxel) => {
        const center = drawVoxel(ctx, layout, voxel, facesFor(voxel));
        const shouldLabel = voxel.label
          && (options.autoLabelDensity === false || layout.unit >= 12 || voxel.state === 'current');
        if (shouldLabel && voxel.state !== 'skipped') {
          ctx.fillStyle = voxel.state === 'current'
            ? cssValue('--background', 'Canvas')
            : cssValue('--foreground', 'CanvasText');
          ctx.fillText(voxel.label, center.x, center.y + 1);
        }
        if (voxel.state === 'skipped') {
          ctx.fillStyle = cssValue('--foreground-muted', 'CanvasText');
          ctx.beginPath();
          ctx.arc(center.x, center.y, 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }

    function schedulePaint() {
      if (destroyed || frame) return;
      frame = global.requestAnimationFrame(() => {
        frame = 0;
        paint();
      });
    }

    if (typeof ResizeObserver === 'function') {
      observer = new ResizeObserver(schedulePaint);
      observer.observe(canvas);
    }

    const controller = {
      update(nextScene, nextOptions) {
        scene = normalizeScene(nextScene);
        if (nextOptions && typeof nextOptions === 'object') {
          options = { ...options, ...nextOptions };
          if (nextOptions.ariaLabel) canvas.setAttribute('aria-label', nextOptions.ariaLabel);
        }
        paint();
        return controller;
      },
      resize() {
        paint();
        return controller;
      },
      destroy() {
        if (destroyed) return;
        destroyed = true;
        observer?.disconnect();
        observer = null;
        if (frame) global.cancelAnimationFrame(frame);
        frame = 0;
      },
    };

    paint();
    return controller;
  }

  global.PtoTensorVolumeCanvas = Object.freeze({ render });
})(window);
