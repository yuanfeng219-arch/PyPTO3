(function () {
  'use strict';

  const RUNS = window.TUNING_RUNS || {};
  const select = document.querySelector('#caseSelect');
  const canvas = document.querySelector('#timeline');
  const shell = document.querySelector('#chartShell');
  const tooltip = document.querySelector('#taskTooltip');
  const state = { caseId: Object.keys(RUNS)[0], filter: 'all', layout: [], selected: null };
  const C = { aic: '#5a8cf5', aiv: '#50b88b', scheduler: '#8b73e6', ready: '#d9a441', muted: '#7e8899', grid: '#303949' };

  const fmtUs = (v) => Number(v || 0).toFixed(v < 100 ? 1 : 0) + ' us';
  const pct = (v) => Number(v || 0).toFixed(0) + '%';
  const current = () => RUNS[state.caseId];
  const rank = () => current().ranks[current().defaultRank];
  const setText = (id, value) => { document.querySelector(id).textContent = value; };
  const phaseList = (r) => Object.entries(r.scheduler.phases).filter(([, v]) => v.us > 0).sort((a, b) => b[1].us - a[1].us);

  function setupCases() {
    Object.values(RUNS).forEach((run) => {
      const option = document.createElement('option');
      option.value = run.case.id;
      option.textContent = run.case.label + ' · ' + run.case.sub;
      select.appendChild(option);
    });
    select.value = state.caseId;
    select.addEventListener('change', () => { state.caseId = select.value; state.selected = null; update(); });
  }

  function updateSummary() {
    const run = current(); const r = rank(); const phases = phaseList(r);
    const queue = r.readyStat;
    setText('#traceSpan', fmtUs(r.swimlane.spanUs));
    setText('#schedulerUtil', pct(r.scheduler.perLaneUtil));
    setText('#aicUtil', pct(r.occupancy.aicUtil));
    setText('#aivUtil', pct(r.occupancy.aivUtil));
    const hasQueue = queue.busyShare.AIC > 10;
    setText('#takeaway', hasQueue ? 'AIC 已就绪但尚未派发' : '调度等待不显著');
    const lead = phases[0] || ['—', { us: 0, tasks: 0, usPerTask: 0 }];
    setText('#phaseTitle', lead[0] + ' 是最重阶段 · ' + fmtUs(lead[1].us));
    setText('#phaseDetail', lead[1].tasks + ' 个任务，平均 ' + (lead[1].usPerTask || 0).toFixed(3) + ' us / 任务。');
    setText('#queueTitle', hasQueue ? 'AIC queue 非空 ' + pct(queue.busyShare.AIC) : 'AIC queue 基本为空');
    setText('#queueDetail', '平均 ' + queue.avg.AIC.toFixed(1) + ' 个 · 峰值 ' + queue.peak.AIC + ' 个 ready 任务。');
    document.querySelector('#chartTitle').textContent = run.case.program + ' · ' + r.rank;
  }

  function fitCanvas(width, height) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    canvas.style.width = width + 'px'; canvas.style.height = height + 'px';
    const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return ctx;
  }
  function drawText(ctx, text, x, y, color, align) { ctx.fillStyle = color; ctx.textAlign = align || 'left'; ctx.fillText(text, x, y); }

  function draw() {
    const r = rank();
    const all = r.swimlane.lanes;
    const lanes = state.filter === 'all' ? all : all.filter((lane) => lane.kind === state.filter);
    const width = Math.max(shell.clientWidth || 900, 760); const label = 86; const right = 18; const plot = width - label - right;
    const rowH = 10, rowGap = 4, groupGap = 10, top = 143;
    const ys = []; let cursor = top;
    lanes.forEach((lane, i) => { ys.push(cursor); cursor += rowH + rowGap; if (i < lanes.length - 1 && lane.kind !== lanes[i + 1].kind) cursor += groupGap; });
    const height = cursor + 22; const ctx = fitCanvas(width, Math.max(height, shell.clientHeight || height));
    const span = r.swimlane.spanUs; const sx = (t) => label + (t / span) * plot;
    state.layout = [];
    ctx.clearRect(0, 0, width, height); ctx.font = '11px ' + getComputedStyle(document.documentElement).getPropertyValue('--font-sans'); ctx.textBaseline = 'middle';
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.setLineDash([2, 5]);
    for (let i = 0; i <= 8; i++) { const x = label + (plot / 8) * i; ctx.beginPath(); ctx.moveTo(x + .5, 16); ctx.lineTo(x + .5, height - 10); ctx.stroke(); drawText(ctx, fmtUs(span / 8 * i), x, 10, C.muted, i === 0 ? 'left' : i === 8 ? 'right' : 'center'); }
    ctx.setLineDash([]);

    drawText(ctx, 'SCHEDULER', 8, 33, C.muted); r.scheduler.lanes.forEach((name, index) => {
      const y = 28 + index * 13; drawText(ctx, name, label - 8, y + 4, C.muted, 'right');
      r.scheduler.blocks[index].forEach((block) => { const x = sx(block[0]); const w = Math.max(1, sx(block[0] + block[1]) - x); ctx.globalAlpha = block[2] === 'complete' ? .95 : block[2] === 'dispatch' ? .68 : .38; ctx.fillStyle = C.scheduler; ctx.fillRect(x, y, w, 8); });
    }); ctx.globalAlpha = 1;

    const readyY = 79, readyH = 33, peak = Math.max(r.readyStat.peak.AIC, r.readyStat.peak.AIV, 1);
    drawText(ctx, 'READY', 8, readyY + readyH / 2, C.muted); ctx.beginPath(); ctx.moveTo(label, readyY + readyH);
    r.readyQueue.forEach((q) => ctx.lineTo(sx(q[0]), readyY + readyH - (q[1] / peak) * readyH)); ctx.lineTo(label + plot, readyY + readyH); ctx.closePath(); ctx.fillStyle = C.ready; ctx.globalAlpha = .35; ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = C.ready; ctx.lineWidth = 1.25; ctx.beginPath(); r.readyQueue.forEach((q, i) => { const x = sx(q[0]); const y = readyY + readyH - (q[1] / peak) * readyH; if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke(); drawText(ctx, 'peak ' + peak, label + plot, readyY + 5, C.muted, 'right');
    ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(0, 124.5); ctx.lineTo(width, 124.5); ctx.stroke();

    lanes.forEach((lane, index) => {
      const y = ys[index]; const original = all.indexOf(lane); const kindColor = lane.kind === 'aic' ? C.aic : C.aiv;
      ctx.fillStyle = index % 2 ? 'rgba(255,255,255,.026)' : 'rgba(255,255,255,.012)'; ctx.fillRect(label, y - 2, plot, rowH + 4);
      drawText(ctx, lane.name, label - 8, y + rowH / 2, lane.util > 55 ? '#c6d0de' : C.muted, 'right');
      r.swimlane.blocks[original].forEach((block) => {
        const x = sx(block[0]); const w = Math.max(.9, sx(block[0] + block[1]) - x); const task = r.tasks[block[2]];
        ctx.fillStyle = kindColor; ctx.globalAlpha = task && task.tag === state.selected ? 1 : .72; ctx.fillRect(x, y, w, rowH);
      }); ctx.globalAlpha = 1; state.layout.push({ y, name: lane.name, index: original, h: rowH });
      if (index < lanes.length - 1 && lane.kind !== lanes[index + 1].kind) { ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(0, y + rowH + (rowGap + groupGap) / 2 + .5); ctx.lineTo(width, y + rowH + (rowGap + groupGap) / 2 + .5); ctx.stroke(); }
    });
  }

  function pick(event) {
    const rect = canvas.getBoundingClientRect(); const x = event.clientX - rect.left; const y = event.clientY - rect.top; const r = rank();
    const row = state.layout.find((item) => y >= item.y - 2 && y <= item.y + item.h + 2); if (!row || x < 86) return null;
    const time = ((x - 86) / (rect.width - 104)) * r.swimlane.spanUs;
    const block = r.swimlane.blocks[row.index].find((b) => time >= b[0] - 3 && time <= b[0] + b[1] + 3); if (!block) return null;
    return { task: r.tasks[block[2]], block, lane: row.name };
  }
  function showSelection(hit) {
    if (!hit || !hit.task) return;
    state.selected = hit.task.tag;
    setText('#selectionTitle', hit.task.callable);
    setText('#selectionDetail', hit.lane + ' · ' + fmtUs(hit.block[0]) + ' → ' + fmtUs(hit.block[0] + hit.block[1]) + ' · 持续 ' + fmtUs(hit.block[1]));
    draw();
  }
  function attachInteractions() {
    canvas.addEventListener('mousemove', (event) => { const hit = pick(event); if (!hit || !hit.task) { tooltip.hidden = true; return; } tooltip.hidden = false; tooltip.innerHTML = '<strong>' + hit.task.callable + '</strong><span>' + hit.lane + ' · ' + fmtUs(hit.block[1]) + '</span>'; tooltip.style.left = Math.min(event.clientX + 14, window.innerWidth - 305) + 'px'; tooltip.style.top = (event.clientY + 14) + 'px'; });
    canvas.addEventListener('mouseleave', () => { tooltip.hidden = true; });
    canvas.addEventListener('click', (event) => showSelection(pick(event)));
    document.querySelectorAll('[data-filter]').forEach((button) => button.addEventListener('click', () => { state.filter = button.dataset.filter; document.querySelectorAll('[data-filter]').forEach((item) => item.classList.toggle('is-active', item === button)); draw(); }));
    document.querySelector('#resetButton').addEventListener('click', () => { state.filter = 'all'; state.selected = null; document.querySelector('[data-filter="all"]').click(); setText('#selectionTitle', '移动到泳道上选择一个任务'); setText('#selectionDetail', '这里仅保留定位该任务所需的信息。'); shell.scrollTop = 0; });
    new ResizeObserver(draw).observe(shell);
  }
  function update() { updateSummary(); draw(); }
  setupCases(); attachInteractions(); update();
}());
