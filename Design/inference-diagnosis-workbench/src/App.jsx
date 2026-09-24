import { useEffect, useRef, useState } from 'react';
import '../../../vendor/pto-design-system/patterns/workbench-shell/pattern.js';
import '../../../vendor/pto-design-system/patterns/ide-frame/pattern.js';
import '../../../vendor/pto-design-system/patterns/swimlane-task/pattern.js';

const evidence = [
  ['运行摘要', 'Decode #019 · 异常', 'active'],
  ['测量契约', 'scope: rank 0 · 204 tokens', ''],
  ['主机侧准备', 'rank table build · 18.27 s', 'bad'],
  ['图回放', 'replay hit · 98.6%', ''],
  ['设备侧计算', 'kernel · 5.41 s', ''],
  ['可复现性', 'seed stable · 3 / 3', ''],
];

const lanes = [
  { name: 'request', kind: 'other', bars: [{ id: 'request', x: 0, w: 2.2, label: 'User request' }] },
  { name: 'orchestrator', kind: 'cpu_sched', bars: [{ id: 'pre', x: 2.2, w: 1.1, label: 'Preprocess' }, { id: 'decode', x: 3.3, w: 21.9, label: 'Decode cycle' }] },
  { name: 'host', kind: 'cpu_sched', bars: [{ id: 'rank', x: 3.6, w: 18.27, label: 'rank table build' }, { id: 'dispatch', x: 22.0, w: 1.0, label: 'dispatch' }] },
  { name: 'graph', kind: 'aic', bars: [{ id: 'replay', x: 5.0, w: 16.1, label: 'graph replay' }] },
  { name: 'GPU', kind: 'aiv', bars: [{ id: 'gpu', x: 7.6, w: 5.41, label: 'kernel execution' }, { id: 'gpu2', x: 14.0, w: 4.1, label: 'attention + MLP' }] },
  { name: 'idle', kind: 'fake', bars: [{ id: 'idle', x: 13.1, w: 8.1, label: 'device idle' }] },
];

function TraceCanvas({ selected, onSelect }) {
  const canvasRef = useRef(null);
  const hostRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    const pattern = window.PtoSwimlaneTaskPattern;
    if (!canvas || !host || !pattern) return undefined;
    const context = canvas.getContext('2d');
    const rowHeight = 33;
    const padding = { left: 8, right: 8, top: 8 };
    const tooltip = pattern.initHoverTooltip({ root: host, targets: [], bounds: host });
    let hovered = null;

    const paint = () => {
      const width = Math.max(320, Math.floor(host.clientWidth));
      const height = padding.top + lanes.length * rowHeight + 8;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      canvas.style.height = `${height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      const styles = getComputedStyle(host);
      const grid = styles.getPropertyValue('--border-subtle').trim() || '#303947';
      const text = styles.getPropertyValue('--fg-muted').trim() || '#9aa4b2';
      const colors = pattern.createTaskColormap();
      context.clearRect(0, 0, width, height);
      lanes.forEach((lane, i) => {
        const y = padding.top + i * rowHeight;
        context.strokeStyle = grid;
        context.lineWidth = 1;
        context.beginPath(); context.moveTo(0, y + rowHeight - 1); context.lineTo(width, y + rowHeight - 1); context.stroke();
        lane.bars.forEach((bar) => {
          const task = { ...bar, lane: lane.name, laneKind: lane.kind, totalCycle: bar.w * 1000 };
          const x = padding.left + (bar.x / 25.3) * (width - padding.left - padding.right);
          const barWidth = Math.max(8, (bar.w / 25.3) * (width - padding.left - padding.right));
          pattern.drawTaskBar(context, { task, x, y: y + 6, width: barWidth, height: 21, baseColor: colors.colorForTask(task, 'engine'), isSelected: selected?.id === bar.id, isEmphasized: hovered?.id === bar.id, fontFamily: styles.getPropertyValue('--font-mono').trim() || 'monospace' });
        });
        context.fillStyle = text;
        context.font = `11px ${styles.getPropertyValue('--font-mono').trim() || 'monospace'}`;
        context.fillText(lane.name, 12, y + 20);
      });
    };
    const locate = (event) => {
      const bounds = canvas.getBoundingClientRect();
      const x = ((event.clientX - bounds.left - padding.left) / (bounds.width - padding.left - padding.right)) * 25.3;
      const row = Math.floor((event.clientY - bounds.top - padding.top) / rowHeight);
      const lane = lanes[row];
      return lane?.bars.find((bar) => x >= bar.x && x <= bar.x + bar.w) || null;
    };
    const onMove = (event) => {
      const bar = locate(event);
      if (bar?.id !== hovered?.id) { hovered = bar; paint(); }
      if (bar) pattern.showTooltip(tooltip.tooltip, { ...bar, lane: lanes.find((lane) => lane.bars.includes(bar))?.name, totalCycle: bar.w * 1000 }, event, { bounds: host, durationUnit: 'ms' });
      else pattern.hideTooltip(tooltip.tooltip);
    };
    const onLeave = () => { hovered = null; pattern.hideTooltip(tooltip.tooltip); paint(); };
    const onClick = (event) => { const bar = locate(event); if (bar) onSelect(bar); };
    const resize = new ResizeObserver(paint);
    resize.observe(host); paint();
    canvas.addEventListener('pointermove', onMove); canvas.addEventListener('pointerleave', onLeave); canvas.addEventListener('click', onClick);
    return () => { resize.disconnect(); tooltip.destroy(); canvas.removeEventListener('pointermove', onMove); canvas.removeEventListener('pointerleave', onLeave); canvas.removeEventListener('click', onClick); };
  }, [selected, onSelect]);

  return <div className="trace-canvas" ref={hostRef}><canvas ref={canvasRef} aria-label="Decode execution swimlane; hover a span for timing evidence" /></div>;
}

function RailButton({ label, active, toggle, children }) {
  return <button className={`btn btn-icon btn-ghost pto-ide-frame__rail-button ${active ? 'is-selected' : ''}`} type="button" data-ide-toggle={toggle} aria-label={label} title={label}>{children}</button>;
}

export function App() {
  const frameRef = useRef(null);
  const [selected, setSelected] = useState(lanes[2].bars[0]);
  const [planCreated, setPlanCreated] = useState(false);
  useEffect(() => window.PtoIdeFrame?.init(frameRef.current)?.refresh(), []);

  return <main className="diagnosis-page" data-theme="dark">
    <section className="pto-ide-frame diagnosis-frame" ref={frameRef} data-ide-frame data-host="standalone" aria-label="PTO3 推理诊断工作台">
      <header className="pto-ide-frame__topbar pto-ide-frame__standalone-chrome">
        <div className="pto-ide-frame__topbar-left"><div className="pto-ide-frame__host-chip">PTO3</div><div className="workspace-label">Runtime Diagnostics <span>/ inference trace</span></div></div>
        <div className="pto-ide-frame__topbar-center"><label className="top-search"><span>⌕</span><input aria-label="搜索运行、算子或证据" placeholder="Search trace, span, artifact…" /></label></div>
        <div className="pto-ide-frame__topbar-right"><button className="btn btn-ghost top-action" type="button">导出证据</button><button className="btn btn-ghost top-action" type="button" data-ide-toggle="inspector" aria-controls="diagnosis-inspector" aria-expanded="true">检查器</button></div>
      </header>
      <div className="pto-ide-frame__body">
        <nav className="pto-ide-frame__activity-rail pto-ide-frame__standalone-chrome" aria-label="Activity rail">
          <RailButton label="Explorer" active toggle="explorer">▤</RailButton><RailButton label="Search">⌕</RailButton><RailButton label="Source control">⌘</RailButton><RailButton label="Terminal">›_</RailButton>
        </nav>
        <div className="pto-ide-frame__workarea">
          <div className="pto-ide-frame__split" data-ide-split="standalone-main" data-split-direction="horizontal" data-sizes="23,50,27" data-min-size="220,420,260">
            <aside className="pto-ide-frame__pane pto-ide-frame__explorer" data-ide-pane="explorer" id="diagnosis-explorer">
              <header className="pto-ide-frame__pane-header"><span className="pto-ide-frame__pane-title">TRACE EVIDENCE</span><span className="pane-meta">1 selected</span></header>
              <div className="pto-ide-frame__pane-body evidence-body">
                <div className="run-card"><span className="status-dot is-bad" /><div><strong>DeepSeek-V4.1-Flash</strong><small>TP16 / EP16 · 2026-09-23 10:41</small></div></div>
                <div className="section-label">异常运行</div>
                <div className="evidence-list">{evidence.map(([title, meta, state]) => <button className={`evidence-item ${state}`} type="button" key={title}><span>{title}</span><small>{meta}</small></button>)}</div>
                <div className="section-label">相似运行</div>
                <div className="related-runs"><button type="button"><span>Decode #018</span><small>TTFT 1.06s · normal</small></button><button type="button"><span>Decode #017</span><small>host prep 0.82s · normal</small></button></div>
              </div>
            </aside>
            <section className="pto-ide-frame__pane trace-main" data-ide-pane="editor-preview">
              <div className="pto-ide-frame__tabstrip"><span className="active-tab">Decode #019</span><span>Host profile</span><span>Contract</span></div>
              <header className="pto-ide-frame__pane-header trace-header"><div><span className="pto-ide-frame__pane-title">Execution timeline</span><span className="pane-meta">rank 0 · decode token 204</span></div><span className="live-badge">captured</span></header>
              <div className="pto-ide-frame__pane-body trace-body">
                <div className="metric-row"><div><small>TTFT</small><strong>1.08 s</strong></div><div><small>Decode throughput</small><strong>8.4 tokens/s</strong></div><div><small>Total wall time</small><strong>25.31 s</strong></div><div><small>Device idle</small><strong className="bad-text">32.0%</strong></div></div>
                <div className="timeline-topline"><span>0 ms</span><span>6.3 s</span><span>12.6 s</span><span>19.0 s</span><span>25.3 s</span></div>
                <TraceCanvas selected={selected} onSelect={setSelected} />
                <div className="legend"><span><i className="legend-host" />host</span><span><i className="legend-graph" />graph replay</span><span><i className="legend-gpu" />GPU</span><span><i className="legend-idle" />idle / waiting</span><span className="selection-text">selected: {selected.label}</span></div>
                <section className="attribution-table"><div className="table-title"><span>Critical path attribution</span><span>scope: Decode #019</span></div><div className="table-row table-head"><span>span</span><span>wall time</span><span>share</span><span>evidence</span></div><div className="table-row emphasized"><span>host rank table build</span><span>18.27 s</span><span>72.2%</span><a href="#evidence">stack sample ↗</a></div><div className="table-row"><span>device kernel execution</span><span>5.41 s</span><span>21.4%</span><a href="#evidence">kernel trace ↗</a></div><div className="table-row"><span>graph replay / dispatch</span><span>1.63 s</span><span>6.4%</span><a href="#evidence">replay log ↗</a></div></section>
              </div>
            </section>
            <aside className="pto-ide-frame__pane inspector-pane" data-ide-pane="inspector" id="diagnosis-inspector">
              <header className="pto-ide-frame__pane-header"><span className="pto-ide-frame__pane-title">诊断结论</span><span className="danger-badge">High confidence</span></header>
              <div className="pto-ide-frame__pane-body inspector-body">
                <p className="eyebrow">Critical path / 72.2%</p><h1>解码延迟主要由主机侧表构建占用</h1><p className="finding">设备执行并非主要矛盾：GPU kernel 仅占 21.4%，且图回放命中稳定。异常集中在每轮 Decode 重建 rank table。</p>
                <div className="diagnosis-block"><span>测量范围</span><strong>rank 0 · Decode #019 · 204 tokens</strong><small>wall clock + host trace + kernel trace</small></div>
                <div className="diagnosis-block"><span>反证检查</span><strong>seed / shape / cache contract 一致</strong><small>3 次重放方差 &lt; 1.8%</small></div>
                <div className="link-stack"><a href="#evidence">查看 host stack sample ↗</a><a href="#evidence">查看 graph replay contract ↗</a><a href="#evidence">与 #018 正常运行对比 ↗</a></div>
                <button className="btn btn-primary plan-button" type="button" onClick={() => setPlanCreated(true)}>{planCreated ? '验证计划已创建' : '创建验证计划'}</button>
                {planCreated && <p className="plan-note">已生成：缓存 rank table · A/B 三轮 Decode · 记录 TTFT 与 idle ratio。</p>}
              </div>
            </aside>
          </div>
          <footer className="pto-ide-frame__status-strip pto-ide-frame__standalone-chrome"><span>● trace contract valid</span><span>rank 0 / world 16</span><span>collector: host+device</span><span>2026-09-23 10:41:32</span></footer>
        </div>
      </div>
    </section>
  </main>;
}
