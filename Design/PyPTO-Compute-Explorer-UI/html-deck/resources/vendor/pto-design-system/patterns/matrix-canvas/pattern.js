(function attachPtoMatrixCanvas(global) {
  'use strict';

  const BASE_CELL = 32;
  const DEFAULT_PADDING = Object.freeze({ top: 38, right: 28, bottom: 42, left: 52 });
  const VALID_TONES = new Set(['neutral', 'input', 'output', 'compute', 'reduction', 'fusion']);
  const VALID_STYLES = new Set(['value', 'empty', 'padding', 'broadcast', 'aggregate']);
  const VALID_STATES = new Set(['row-focus', 'column-focus', 'written', 'selected', 'current', 'muted']);
  const TONE_TOKENS = Object.freeze({
    neutral: '--foreground-secondary',
    input: '--primary',
    output: '--success',
    compute: '--warning',
    reduction: '--warning',
    fusion: '--accent',
  });

  function positiveInteger(value, fallback = 1) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? Math.floor(number) : fallback;
  }

  function finiteOr(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function resolvePadding(input) {
    const padding = input && typeof input === 'object' ? input : {};
    return {
      top: Math.max(0, finiteOr(padding.top, DEFAULT_PADDING.top)),
      right: Math.max(0, finiteOr(padding.right, DEFAULT_PADDING.right)),
      bottom: Math.max(0, finiteOr(padding.bottom, DEFAULT_PADDING.bottom)),
      left: Math.max(0, finiteOr(padding.left, DEFAULT_PADDING.left)),
    };
  }

  function cssValue(name, fallback) {
    const value = global.getComputedStyle?.(document.documentElement)?.getPropertyValue(name)?.trim();
    return value || fallback;
  }

  function parseCssColor(value) {
    const input = String(value || '').trim();
    const shortHex = /^#([0-9a-f]{3})$/i.exec(input);
    if (shortHex) return shortHex[1].split('').map((part) => parseInt(`${part}${part}`, 16));
    const longHex = /^#([0-9a-f]{6})(?:[0-9a-f]{2})?$/i.exec(input);
    if (longHex) {
      const number = parseInt(longHex[1], 16);
      return [(number >> 16) & 255, (number >> 8) & 255, number & 255];
    }
    const rgb = /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i.exec(input);
    return rgb ? rgb.slice(1, 4).map(Number) : null;
  }

  function resolvedTokenRgb(name, visited = new Set()) {
    if (!name || visited.has(name)) return null;
    visited.add(name);
    const value = cssValue(name, '');
    const direct = parseCssColor(value);
    if (direct) return direct;
    const reference = /var\(\s*(--[\w-]+)/.exec(value);
    return reference ? resolvedTokenRgb(reference[1], visited) : null;
  }

  function tokenRgb(name, fallback = '--foreground-secondary') {
    return resolvedTokenRgb(name) || resolvedTokenRgb(fallback) || [128, 128, 128];
  }

  function rgba(rgb, alpha = 1) {
    const channels = rgb.map((value) => clamp(Math.round(value), 0, 255));
    return `rgb(${channels[0]} ${channels[1]} ${channels[2]} / ${clamp(alpha, 0, 1)})`;
  }

  function mix(source, target, sourceWeight) {
    const weight = clamp(sourceWeight, 0, 1);
    return source.map((value, index) => value * weight + target[index] * (1 - weight));
  }

  function normalizeScene(input) {
    const scene = input && typeof input === 'object' ? input : {};
    const extent = scene.extent && typeof scene.extent === 'object' ? scene.extent : {};
    return {
      extent: {
        rows: positiveInteger(extent.rows),
        columns: positiveInteger(extent.columns),
      },
      axes: {
        rows: String(scene.axes?.rows || 'row'),
        columns: String(scene.axes?.columns || 'column'),
      },
      cells: Array.isArray(scene.cells) ? scene.cells : [],
    };
  }

  function normalizeSummary(input, rowSpan, columnSpan) {
    const summary = input && typeof input === 'object' ? input : {};
    const samples = Array.isArray(summary.samples) || ArrayBuffer.isView(summary.samples)
      ? Array.from(summary.samples, (value) => Number(value))
      : [];
    return {
      rows: positiveInteger(summary.rows, rowSpan),
      columns: positiveInteger(summary.columns, columnSpan),
      count: positiveInteger(summary.count, rowSpan * columnSpan),
      min: finiteOr(summary.min, NaN),
      max: finiteOr(summary.max, NaN),
      mean: finiteOr(summary.mean, NaN),
      intensity: clamp(finiteOr(summary.intensity, 0.5), 0, 1),
      samples: samples.filter(Number.isFinite),
    };
  }

  function normalizeCell(input, extent, index) {
    if (!input || typeof input !== 'object') {
      console.warn(`[PtoMatrixCanvas] Skipped cell ${index}: expected an object.`);
      return null;
    }
    const row = Math.floor(Number(input.row));
    const column = Math.floor(Number(input.column));
    if (!Number.isFinite(row) || !Number.isFinite(column) || row < 0 || column < 0
      || row >= extent.rows || column >= extent.columns) {
      console.warn(`[PtoMatrixCanvas] Skipped cell ${input.id || index}: coordinate is outside scene extent.`);
      return null;
    }
    const rowSpan = Math.min(positiveInteger(input.rowSpan), extent.rows - row);
    const columnSpan = Math.min(positiveInteger(input.columnSpan), extent.columns - column);
    const states = new Set(
      (Array.isArray(input.states) ? input.states : [input.state])
        .filter((state) => VALID_STATES.has(state))
    );
    const style = VALID_STYLES.has(input.style) ? input.style : 'value';
    return {
      id: String(input.id || `cell-${row}-${column}`),
      row,
      column,
      rowSpan,
      columnSpan,
      label: input.label == null ? '' : String(input.label),
      value: finiteOr(input.value, NaN),
      tone: VALID_TONES.has(input.tone) ? input.tone : 'neutral',
      style,
      states,
      summary: style === 'aggregate' ? normalizeSummary(input.summary, rowSpan, columnSpan) : null,
      source: input,
    };
  }

  function roundedRect(ctx, x, y, width, height, radius) {
    const resolved = clamp(radius, 0, Math.min(width, height) / 2);
    ctx.beginPath();
    ctx.moveTo(x + resolved, y);
    ctx.arcTo(x + width, y, x + width, y + height, resolved);
    ctx.arcTo(x + width, y + height, x, y + height, resolved);
    ctx.arcTo(x, y + height, x, y, resolved);
    ctx.arcTo(x, y, x + width, y, resolved);
    ctx.closePath();
  }

  function fitView(width, height, extent, padding) {
    const innerWidth = Math.max(1, width - padding.left - padding.right);
    const innerHeight = Math.max(1, height - padding.top - padding.bottom);
    const worldWidth = extent.columns * BASE_CELL;
    const worldHeight = extent.rows * BASE_CELL;
    const scale = Math.min(innerWidth / worldWidth, innerHeight / worldHeight, 1.6);
    return {
      scale,
      offsetX: padding.left + (innerWidth - worldWidth * scale) / 2,
      offsetY: padding.top + (innerHeight - worldHeight * scale) / 2,
    };
  }

  function cellScreenRect(cell, view) {
    return {
      x: view.offsetX + cell.column * BASE_CELL * view.scale,
      y: view.offsetY + cell.row * BASE_CELL * view.scale,
      width: cell.columnSpan * BASE_CELL * view.scale,
      height: cell.rowSpan * BASE_CELL * view.scale,
    };
  }

  function visible(rect, width, height) {
    return rect.x + rect.width >= 0 && rect.y + rect.height >= 0 && rect.x <= width && rect.y <= height;
  }

  function cellColors(cell) {
    const background = tokenRgb('--background');
    const foreground = tokenRgb('--foreground');
    const surface2 = tokenRgb('--surface-2');
    const surface3 = tokenRgb('--surface-3');
    const tone = tokenRgb(TONE_TOKENS[cell.tone]);
    let fill = rgba(mix(surface2, background, 0.82), 0.96);
    let stroke = rgba(foreground, 0.09);
    let text = rgba(foreground, 0.68);

    if (cell.style === 'empty') {
      fill = rgba(surface2, 0.32);
      text = rgba(foreground, 0.38);
    } else if (cell.style === 'aggregate') {
      fill = rgba(mix(surface3, background, 0.76), 0.98);
      stroke = rgba(foreground, 0.14);
    } else if (cell.tone !== 'neutral') {
      fill = rgba(mix(tone, background, 0.22), 0.96);
    }
    if (cell.states.has('row-focus')) fill = rgba(mix(tokenRgb('--primary'), background, 0.2), 0.98);
    if (cell.states.has('column-focus')) fill = rgba(mix(tokenRgb('--warning'), parseCssColor(fill) || background, 0.12), 0.98);
    if (cell.states.has('written')) {
      fill = rgba(mix(tokenRgb('--primary'), background, 0.34), 0.98);
      text = rgba(foreground, 0.78);
    }
    if (cell.states.has('muted')) {
      fill = rgba(surface2, 0.25);
      text = rgba(foreground, 0.35);
    }
    if (cell.states.has('selected')) {
      stroke = rgba(foreground, 0.92);
      text = rgba(foreground, 0.92);
      if (cell.style === 'aggregate') {
        fill = rgba(tokenRgb('--primary'), 1);
        stroke = rgba(tokenRgb('--primary-hover'), 1);
        text = rgba(foreground, 0.92);
      }
    }
    if (cell.states.has('current')) {
      fill = rgba(tokenRgb('--warning'), 1);
      stroke = rgba(mix(tokenRgb('--warning'), foreground, 0.66), 1);
      text = rgba(background, 0.92);
    }
    return { fill, stroke, text, tone, background, foreground };
  }

  function drawHatch(ctx, rect) {
    const foreground = tokenRgb('--foreground');
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x, rect.y, rect.width, rect.height);
    ctx.clip();
    ctx.strokeStyle = rgba(foreground, 0.1);
    ctx.lineWidth = 1;
    const step = clamp(Math.min(rect.width, rect.height) * 0.3, 6, 12);
    for (let start = rect.x - rect.height; start < rect.x + rect.width; start += step) {
      ctx.beginPath();
      ctx.moveTo(start, rect.y + rect.height);
      ctx.lineTo(start + rect.height, rect.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawAggregate(ctx, cell, rect, colors) {
    const minDimension = Math.min(rect.width, rect.height);
    if (minDimension < 7) return;
    const markerSize = clamp(minDimension * 0.22, 4, 28);
    const markerX = rect.x + (rect.width - markerSize) / 2;
    const markerY = rect.y + (rect.height - markerSize) / 2;
    const selected = cell.states.has('selected') || cell.states.has('current');
    const intensity = clamp(cell.summary?.intensity ?? 0.5, 0, 1);
    const markerColor = selected
      ? colors.foreground
      : cell.tone === 'neutral'
        ? tokenRgb('--foreground')
        : colors.tone;
    roundedRect(ctx, markerX, markerY, markerSize, markerSize, clamp(markerSize * 0.16, 1.5, 5));
    ctx.fillStyle = rgba(markerColor, selected ? 0.24 : 0.18 + intensity * 0.34);
    ctx.fill();
  }

  function fittedText(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) return text;
    let output = text;
    while (output.length > 1 && ctx.measureText(`${output}…`).width > maxWidth) output = output.slice(0, -1);
    return output ? `${output}…` : '';
  }

  function drawCell(ctx, cell, rect) {
    const colors = cellColors(cell);
    const minDimension = Math.min(rect.width, rect.height);
    const box = {
      x: rect.x,
      y: rect.y,
      width: Math.max(0.2, rect.width),
      height: Math.max(0.2, rect.height),
    };

    ctx.fillStyle = colors.fill;
    ctx.fillRect(box.x, box.y, box.width, box.height);
    if (cell.style === 'broadcast' && !cell.states.has('current')) {
      const grayMask = mix(tokenRgb('--foreground-secondary'), colors.background, 0.4);
      ctx.fillStyle = rgba(grayMask, 0.4);
      ctx.fillRect(box.x, box.y, box.width, box.height);
    }
    ctx.save();
    ctx.strokeStyle = colors.stroke;
    ctx.lineWidth = cell.states.has('current') ? 2 : cell.states.has('selected') ? 1.5 : 1;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.restore();

    if (cell.style === 'padding') drawHatch(ctx, box);
    if (cell.style === 'aggregate') drawAggregate(ctx, cell, box, colors);

    const mono = cssValue('--font-mono', 'ui-monospace, monospace');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = colors.text;

    if (cell.style === 'aggregate') {
      return;
    }

    if (!cell.label || (minDimension < 40 && !cell.states.has('current'))) return;
    const fontSize = clamp(minDimension * 0.3, 11, 14);
    ctx.font = `700 ${fontSize}px ${mono}`;
    const label = fittedText(ctx, cell.label, Math.max(0, box.width - 8));
    if (label) ctx.fillText(label, box.x + box.width / 2, box.y + box.height / 2);
  }

  function createAggregatedCells(source, inputOptions = {}) {
    if (!source || typeof source !== 'object') {
      throw new TypeError('PtoMatrixCanvas.createAggregatedCells expects a source object.');
    }
    const rows = positiveInteger(source.rows);
    const columns = positiveInteger(source.columns);
    const values = source.values;
    const valueAt = typeof source.valueAt === 'function'
      ? source.valueAt
      : (row, column) => values?.[row * columns + column];
    if (!values && typeof source.valueAt !== 'function') {
      throw new TypeError('Aggregated matrix source requires values or valueAt(row, column).');
    }
    const requestedThumbnailRows = Number(inputOptions.thumbnailRows);
    const requestedThumbnailColumns = Number(inputOptions.thumbnailColumns);
    const blockRows = Number.isFinite(requestedThumbnailRows) && requestedThumbnailRows > 0
      ? Math.ceil(rows / Math.floor(requestedThumbnailRows))
      : positiveInteger(inputOptions.blockRows, 16);
    const blockColumns = Number.isFinite(requestedThumbnailColumns) && requestedThumbnailColumns > 0
      ? Math.ceil(columns / Math.floor(requestedThumbnailColumns))
      : positiveInteger(inputOptions.blockColumns, 16);
    const cells = [];

    for (let row = 0; row < rows; row += blockRows) {
      const rowSpan = Math.min(blockRows, rows - row);
      for (let column = 0; column < columns; column += blockColumns) {
        const columnSpan = Math.min(blockColumns, columns - column);
        let min = Infinity;
        let max = -Infinity;
        let sum = 0;
        let count = 0;

        for (let localRow = 0; localRow < rowSpan; localRow += 1) {
          for (let localColumn = 0; localColumn < columnSpan; localColumn += 1) {
            const value = Number(valueAt(row + localRow, column + localColumn));
            if (!Number.isFinite(value)) continue;
            min = Math.min(min, value);
            max = Math.max(max, value);
            sum += value;
            count += 1;
          }
        }

        cells.push({
          id: `aggregate-${row}-${column}`,
          row,
          column,
          rowSpan,
          columnSpan,
          style: 'aggregate',
          tone: VALID_TONES.has(inputOptions.tone) ? inputOptions.tone : 'neutral',
          summary: {
            rows: rowSpan,
            columns: columnSpan,
            count,
            min: count ? min : NaN,
            max: count ? max : NaN,
            mean: count ? sum / count : NaN,
          },
        });
      }
    }
    const means = cells.map((cell) => cell.summary.mean).filter(Number.isFinite);
    const minimumMean = means.length ? Math.min(...means) : 0;
    const maximumMean = means.length ? Math.max(...means) : 1;
    const meanRange = maximumMean - minimumMean || 1;
    cells.forEach((cell) => {
      cell.summary.intensity = Number.isFinite(cell.summary.mean)
        ? clamp((cell.summary.mean - minimumMean) / meanRange, 0, 1)
        : 0.5;
    });
    return cells;
  }

  function render(canvas, inputScene, inputOptions = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError('PtoMatrixCanvas.render expects an HTMLCanvasElement.');
    }

    let scene = normalizeScene(inputScene);
    let options = { ...inputOptions };
    let cells = scene.cells.map((cell, index) => normalizeCell(cell, scene.extent, index)).filter(Boolean);
    let view = { scale: 1, offsetX: 0, offsetY: 0 };
    let userView = false;
    let needsFit = options.autoFit !== false;
    let observer = null;
    let frame = 0;
    let destroyed = false;
    let hovered = null;
    let pointer = null;

    const host = canvas.parentElement || document.body;
    host.classList.add('pto-matrix-canvas-host');
    const tooltip = document.createElement('div');
    tooltip.className = 'pto-matrix-canvas__tooltip';
    tooltip.setAttribute('role', 'tooltip');
    const tooltipTitle = document.createElement('div');
    tooltipTitle.className = 'pto-matrix-canvas__tooltip-title';
    const tooltipMeta = document.createElement('div');
    tooltipMeta.className = 'pto-matrix-canvas__tooltip-meta';
    tooltip.append(tooltipTitle, tooltipMeta);
    const liveStatus = document.createElement('div');
    liveStatus.className = 'pto-matrix-canvas__sr-status';
    liveStatus.setAttribute('aria-live', 'polite');
    host.append(tooltip, liveStatus);

    canvas.classList.add('pto-matrix-canvas');
    canvas.tabIndex = canvas.tabIndex >= 0 ? canvas.tabIndex : 0;
    canvas.setAttribute('role', 'application');
    canvas.setAttribute('aria-label', options.ariaLabel || 'Two-dimensional matrix tensor');
    canvas.setAttribute(
      'aria-description',
      'Use the wheel to scroll vertically, Shift plus wheel or a horizontal trackpad gesture to scroll horizontally, Control or Command plus wheel to zoom, and press F or 0 to fit.'
    );

    function dimensions() {
      const rect = canvas.getBoundingClientRect();
      return {
        width: Math.max(1, rect.width || canvas.clientWidth || 1),
        height: Math.max(1, rect.height || canvas.clientHeight || 1),
      };
    }

    function applyFit() {
      const { width, height } = dimensions();
      view = fitView(width, height, scene.extent, resolvePadding(options.padding));
      const minZoom = finiteOr(options.minZoom, 0.015);
      const maxZoom = finiteOr(options.maxZoom, 12);
      view.scale = clamp(view.scale, minZoom, maxZoom);
      constrainView();
      needsFit = false;
    }

    function scrollBounds() {
      const { width, height } = dimensions();
      const padding = resolvePadding(options.padding);
      const matrixWidth = scene.extent.columns * BASE_CELL * view.scale;
      const matrixHeight = scene.extent.rows * BASE_CELL * view.scale;
      const innerWidth = Math.max(1, width - padding.left - padding.right);
      const innerHeight = Math.max(1, height - padding.top - padding.bottom);
      const centeredX = padding.left + (innerWidth - matrixWidth) / 2;
      const centeredY = padding.top + (innerHeight - matrixHeight) / 2;
      return {
        minX: matrixWidth <= innerWidth ? centeredX : width - padding.right - matrixWidth,
        maxX: matrixWidth <= innerWidth ? centeredX : padding.left,
        minY: matrixHeight <= innerHeight ? centeredY : height - padding.bottom - matrixHeight,
        maxY: matrixHeight <= innerHeight ? centeredY : padding.top,
      };
    }

    function constrainView() {
      const bounds = scrollBounds();
      view.offsetX = clamp(view.offsetX, bounds.minX, bounds.maxX);
      view.offsetY = clamp(view.offsetY, bounds.minY, bounds.maxY);
      return bounds;
    }

    function matrixBounds() {
      return {
        x: view.offsetX,
        y: view.offsetY,
        width: scene.extent.columns * BASE_CELL * view.scale,
        height: scene.extent.rows * BASE_CELL * view.scale,
      };
    }

    function drawGrid(ctx, bounds, width, height) {
      if (options.showGrid === false) return;
      const spacing = BASE_CELL * view.scale;
      if (spacing < 5 || spacing > 180) return;
      const firstColumn = clamp(Math.floor((0 - view.offsetX) / spacing), 0, scene.extent.columns);
      const lastColumn = clamp(Math.ceil((width - view.offsetX) / spacing), 0, scene.extent.columns);
      const firstRow = clamp(Math.floor((0 - view.offsetY) / spacing), 0, scene.extent.rows);
      const lastRow = clamp(Math.ceil((height - view.offsetY) / spacing), 0, scene.extent.rows);
      ctx.strokeStyle = rgba(tokenRgb('--foreground'), 0.07);
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let column = firstColumn; column <= lastColumn; column += 1) {
        const x = view.offsetX + column * spacing;
        ctx.moveTo(x, Math.max(0, bounds.y));
        ctx.lineTo(x, Math.min(height, bounds.y + bounds.height));
      }
      for (let row = firstRow; row <= lastRow; row += 1) {
        const y = view.offsetY + row * spacing;
        ctx.moveTo(Math.max(0, bounds.x), y);
        ctx.lineTo(Math.min(width, bounds.x + bounds.width), y);
      }
      ctx.stroke();
    }

    function drawAxes(ctx, bounds) {
      if (options.showAxes === false) return;
      const foreground = tokenRgb('--foreground-secondary');
      const mono = cssValue('--font-mono', 'ui-monospace, monospace');
      ctx.fillStyle = rgba(foreground, 0.88);
      ctx.font = `600 12px ${mono}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(scene.axes.columns, bounds.x + bounds.width / 2, bounds.y + bounds.height + 26);
      ctx.save();
      ctx.translate(bounds.x - 28, bounds.y + bounds.height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(scene.axes.rows, 0, 0);
      ctx.restore();
    }

    function paint() {
      if (destroyed) return;
      const { width, height } = dimensions();
      const dpr = Math.max(1, global.devicePixelRatio || 1);
      const backingWidth = Math.max(1, Math.floor(width * dpr));
      const backingHeight = Math.max(1, Math.floor(height * dpr));
      if (canvas.width !== backingWidth) canvas.width = backingWidth;
      if (canvas.height !== backingHeight) canvas.height = backingHeight;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      if (needsFit) applyFit();
      else constrainView();

      const bounds = matrixBounds();
      ctx.save();
      roundedRect(ctx, bounds.x, bounds.y, bounds.width, bounds.height, 4);
      ctx.clip();
      ctx.fillStyle = rgba(tokenRgb('--surface-1'), 0.68);
      ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
      drawGrid(ctx, bounds, width, height);
      cells.forEach((cell) => {
        const rect = cellScreenRect(cell, view);
        if (visible(rect, width, height)) drawCell(ctx, cell, rect);
      });
      ctx.restore();

      roundedRect(ctx, bounds.x, bounds.y, bounds.width, bounds.height, 4);
      ctx.strokeStyle = rgba(tokenRgb('--foreground'), 0.14);
      ctx.lineWidth = 1;
      ctx.stroke();
      drawAxes(ctx, bounds);
    }

    function schedulePaint() {
      if (destroyed || frame) return;
      frame = global.requestAnimationFrame(() => {
        frame = 0;
        paint();
      });
    }

    function eventPoint(event) {
      const rect = canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    function hitTest(point) {
      const worldColumn = (point.x - view.offsetX) / (BASE_CELL * view.scale);
      const worldRow = (point.y - view.offsetY) / (BASE_CELL * view.scale);
      if (worldColumn < 0 || worldRow < 0
        || worldColumn >= scene.extent.columns || worldRow >= scene.extent.rows) return null;
      for (let index = cells.length - 1; index >= 0; index -= 1) {
        const cell = cells[index];
        if (worldColumn >= cell.column && worldColumn < cell.column + cell.columnSpan
          && worldRow >= cell.row && worldRow < cell.row + cell.rowSpan) return cell;
      }
      return null;
    }

    function tooltipCopy(cell) {
      if (!cell) return null;
      const range = cell.rowSpan > 1 || cell.columnSpan > 1
        ? `rows ${cell.row}–${cell.row + cell.rowSpan - 1} · columns ${cell.column}–${cell.column + cell.columnSpan - 1}`
        : `row ${cell.row} · column ${cell.column}`;
      if (cell.style !== 'aggregate') {
        const value = Number.isFinite(cell.value) ? ` · value ${cell.value}` : '';
        const style = cell.style !== 'value' ? ` · style ${cell.style}` : '';
        return { title: cell.label || cell.id, meta: `${range}${value}${style}` };
      }
      const summary = cell.summary;
      const stats = [
        `count ${summary.count}`,
        Number.isFinite(summary.min) ? `min ${Number(summary.min).toPrecision(4)}` : null,
        Number.isFinite(summary.max) ? `max ${Number(summary.max).toPrecision(4)}` : null,
        Number.isFinite(summary.mean) ? `mean ${Number(summary.mean).toPrecision(4)}` : null,
      ].filter(Boolean).join(' · ');
      return { title: `${summary.rows}×${summary.columns} aggregate`, meta: `${range} · ${stats}` };
    }

    function showTooltip(cell, point) {
      if (!cell || options.showTooltip === false) {
        tooltip.classList.remove('is-visible');
        tooltip.style.transform = 'translate(-9999px, -9999px)';
        return;
      }
      const copy = tooltipCopy(cell);
      tooltipTitle.textContent = copy.title;
      tooltipMeta.textContent = copy.meta;
      liveStatus.textContent = `${copy.title}. ${copy.meta}`;
      tooltip.classList.add('is-visible');
      const hostRect = host.getBoundingClientRect();
      const canvasRect = canvas.getBoundingClientRect();
      const localX = canvasRect.left - hostRect.left + point.x + 14;
      const localY = canvasRect.top - hostRect.top + point.y + 14;
      const maxX = Math.max(8, host.clientWidth - tooltip.offsetWidth - 8);
      const maxY = Math.max(8, host.clientHeight - tooltip.offsetHeight - 8);
      tooltip.style.transform = `translate(${clamp(localX, 8, maxX)}px, ${clamp(localY, 8, maxY)}px)`;
    }

    function setHovered(next, event) {
      if (hovered?.id === next?.id) {
        if (next && pointer) showTooltip(next, pointer);
        return;
      }
      hovered = next;
      showTooltip(next, pointer || { x: 0, y: 0 });
      options.onHover?.(next ? next.source : null, event);
    }

    function setZoom(nextScale, anchor) {
      const minZoom = finiteOr(options.minZoom, 0.015);
      const maxZoom = finiteOr(options.maxZoom, 12);
      const scale = clamp(finiteOr(nextScale, view.scale), minZoom, maxZoom);
      const point = anchor || { x: dimensions().width / 2, y: dimensions().height / 2 };
      const worldX = (point.x - view.offsetX) / view.scale;
      const worldY = (point.y - view.offsetY) / view.scale;
      view.offsetX = point.x - worldX * scale;
      view.offsetY = point.y - worldY * scale;
      view.scale = scale;
      constrainView();
      userView = true;
      needsFit = false;
      paint();
    }

    function onPointerMove(event) {
      pointer = eventPoint(event);
      setHovered(hitTest(pointer), event);
    }

    function onClick(event) {
      if (options.interactive === false) return;
      pointer = eventPoint(event);
      const cell = hitTest(pointer);
      setHovered(cell, event);
      options.onSelect?.(cell ? cell.source : null, event);
    }

    function onPointerLeave(event) {
      pointer = null;
      setHovered(null, event);
    }

    function onWheel(event) {
      if (options.interactive === false) return;
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        const point = eventPoint(event);
        const factor = Math.exp(-event.deltaY * 0.0014);
        setZoom(view.scale * factor, point);
        return;
      }

      // Let the browser handle normal page scrolling.
      // The matrix canvas only zooms when Ctrl/Cmd is pressed.
      return;
    }

    function onKeyDown(event) {
      if (options.interactive === false) return;
      if (event.key === 'f' || event.key === 'F' || event.key === '0') {
        event.preventDefault();
        controller.fit();
      } else if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        setZoom(view.scale * 1.18);
      } else if (event.key === '-' || event.key === '_') {
        event.preventDefault();
        setZoom(view.scale / 1.18);
      }
    }

    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerleave', onPointerLeave);
    canvas.addEventListener('click', onClick);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('keydown', onKeyDown);

    if (typeof ResizeObserver === 'function') {
      observer = new ResizeObserver(() => {
        if (!userView && options.autoFit !== false) needsFit = true;
        schedulePaint();
      });
      observer.observe(canvas);
    }

    const controller = {
      update(nextScene, nextOptions) {
        const previousExtent = scene.extent;
        scene = normalizeScene(nextScene);
        options = nextOptions && typeof nextOptions === 'object' ? { ...options, ...nextOptions } : options;
        cells = scene.cells.map((cell, index) => normalizeCell(cell, scene.extent, index)).filter(Boolean);
        if (nextOptions?.ariaLabel) canvas.setAttribute('aria-label', nextOptions.ariaLabel);
        const extentChanged = previousExtent.rows !== scene.extent.rows
          || previousExtent.columns !== scene.extent.columns;
        if (extentChanged && nextOptions?.preserveView !== true) {
          userView = false;
          needsFit = true;
        }
        hovered = null;
        showTooltip(null);
        paint();
        return controller;
      },
      resize() {
        if (!userView && options.autoFit !== false) needsFit = true;
        paint();
        return controller;
      },
      fit() {
        userView = false;
        needsFit = true;
        paint();
        return controller;
      },
      resetView() {
        return controller.fit();
      },
      setZoom(nextZoom, anchor) {
        setZoom(nextZoom, anchor);
        return controller;
      },
      getViewState() {
        return Object.freeze({ ...view, userView });
      },
      destroy() {
        if (destroyed) return;
        destroyed = true;
        observer?.disconnect();
        observer = null;
        if (frame) global.cancelAnimationFrame(frame);
        frame = 0;
        canvas.removeEventListener('pointermove', onPointerMove);
        canvas.removeEventListener('pointerleave', onPointerLeave);
        canvas.removeEventListener('click', onClick);
        canvas.removeEventListener('wheel', onWheel);
        canvas.removeEventListener('keydown', onKeyDown);
        tooltip.remove();
        liveStatus.remove();
      },
    };

    paint();
    return controller;
  }

  global.PtoMatrixCanvas = Object.freeze({ render, createAggregatedCells });
})(window);
