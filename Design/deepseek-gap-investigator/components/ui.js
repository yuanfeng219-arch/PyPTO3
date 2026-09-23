/* Shared compositions: PTO controls and surfaces, shadcn/ui table/dialog adapters. */
window.InvestigationUI = (() => {
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const paths = {
    activity:'M3 12h4l3-8 4 16 3-8h4', search:'m21 21-4.3-4.3 M19 11a8 8 0 1 1-16 0 8 8 0 0 1 16 0',
    layers:'m12 3 10 5-10 5L2 8Z M2 12l10 5 10-5 M2 16l10 5 10-5',
    branch:'M6 3v12a5 5 0 0 0 10 0V9 M6 9h5a5 5 0 0 0 5-5',
    sliders:'M4 21v-7 M4 10V3 M12 21v-9 M12 8V3 M20 21v-5 M20 12V3 M1 14h6 M9 8h6 M17 16h6',
    flask:'M9 3h6 M10 3v6L4 19q-1 2 2 2h12q3 0 2-2L14 9V3 M8 14h8',
    clock:'M12 8v4l3 2 M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
    file:'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z M14 2v6h6 M8 13h8 M8 17h5',
    alert:'M12 8v5 M12 16h.01 M10 3 2 18q-1 2 2 2h16q3 0 2-2L14 3q-2-3-4 0',
    arrow:'M4 12h16 M14 6l6 6-6 6', check:'m5 12 4 4L19 6', info:'M12 11v6 M12 7h.01 M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
    database:'M20 5c0 4-16 4-16 0s16-4 16 0v14c0 4-16 4-16 0V5 M4 12c0 4 16 4 16 0',
    terminal:'M4 4h16v16H4Z M7 8l3 3-3 3 M12 15h5', panel:'M3 4h18v16H3Z M15 4v16',
    code:'m8 6-6 6 6 6 M16 6l6 6-6 6',plus:'M12 5v14 M5 12h14'
  };
  const icon = n => `<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${paths[n] || paths.info}"/></svg>`;
  const statusLabels = {
    CONFIRMED:'已确认', SUPPORTED:'有数据支持', CANDIDATE:'待验证', UNKNOWN:'暂无法判断',
    HYPOTHESIS:'待验证方案', DERIVED:'计算结果', RUNTIME:'Trace 记录', COMPILED:'编译产物'
  };
  const badge = (t,tone='neutral') => {
    const label=String(t??'').replace(/CONFIRMED|SUPPORTED|CANDIDATE|UNKNOWN|HYPOTHESIS|DERIVED|RUNTIME|COMPILED/g,key=>statusLabels[key]);
    return `<span class="stat-chip ui-badge ${tone}">${esc(label)}</span>`;
  };
  const button = (l,a,v='',x='') => `<button type="button" class="btn ${v}" data-action="${esc(a)}" ${x}>${l}</button>`;
  const card = (t,b,m='',x='') => `<section class="card-demo ui-card ${x}"><header class="card-demo-header"><div class="section-head"><h3>${t}</h3>${m}</div></header><div class="card-demo-content">${b}</div></section>`;
  const table = (hs,rs,c='') => `<div class="ui-table-wrap"><table class="ui-table">${c?`<caption>${c}</caption>`:''}<thead><tr>${hs.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rs.map(r=>`<tr>${r.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const empty = (t,d,a='') => `<div class="empty-card"><h3 class="empty-title">${t}</h3><p class="empty-sub">${d}</p><div class="empty-actions">${a}</div></div>`;
  const notice = (t,b,tone='warning',a='') => `<div class="inspector-soft-card is-${tone}" role="note"><strong>${t}</strong><p>${b}</p>${a}</div>`;
  const code = s => `<div class="code-view">${s.lines.map((l,i)=>`<div class="code-row ${i+s.start===s.line?'is-highlighted':''}"><span>${i+s.start}</span><code>${esc(l)}</code></div>`).join('')}</div>`;
  return {esc,icon,badge,button,card,table,empty,notice,code};
})();
