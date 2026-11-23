/* 工作台嵌在 Operator Lens 标签页（lens.html）里时的外壳调整：只在 workbench.html?embed=lens 下生效，单独打开工作台不受影响。
 *   1. 去掉工作台自己的顶栏（返回键、品牌、路径、L1 / L2、详情、配置点、编译并上板），这些由外层标签页的导航栏承担；
 *   2. 层级与详情与外层同步：外层发 { type:'lens:go', level } 切 L1 / L2、{ type:'lens:detail' } 开关详情，
 *      本页在层级 / 路径 / 详情变化时回报 { type:'lens:level', level, crumb, detail }；
 *      用 postMessage，file:// 下各文件互为跨源也能工作；
 *   3. 离页链接（返回首页、编译并上板……）在整页打开，不留在 iframe 里。 */
(function () {
  'use strict';
  if (new URLSearchParams(location.search).get('embed') !== 'lens') return;
  document.body.classList.add('is-embedded');
  if (new URLSearchParams(location.search).has('transition')) document.body.classList.add('source-in-flight');

  const css = document.createElement('style');
  css.textContent =
    'body.is-embedded .hp-top { display:none !important; }\n' +
    'body.is-embedded .hp-app { grid-template-rows:minmax(0,1fr) 28px; }\n' +
    '.agent-source-preview { position:absolute; z-index:3; left:0; right:0; overflow:auto; background:var(--canvas); padding:8px 0 100vh; font:12px/1.75 var(--mono); }\n' +
    '.agent-source-preview .cl { line-height:22px; }\n' +
    '.agent-source-preview .cl.is-add .cl-mark { box-shadow:inset 2px 0 0 color-mix(in srgb,var(--lime) 70%,transparent); }\n' +
    '@media(max-width:1280px) and (min-width:861px){ body.is-embedded .hp-app { grid-template-rows:auto 28px; } }\n' +
    '@media(max-width:860px){ body.is-embedded .hp-app { grid-template-rows:auto auto; } }';
  document.head.appendChild(css);

  document.addEventListener('click', function (e) {
    const a = e.target.closest && e.target.closest('a[href]');
    if (!a || a.target) return;
    const h = a.getAttribute('href') || '';
    if (!h || h.charAt(0) === '#') return;
    a.target = '_top';
  }, true);

  let last = '';
  function report() {
    const level = Number(document.body.dataset.level) || 1;
    const cur = document.querySelector('#lensCrumb [aria-current="true"]') || document.querySelector('#lensCrumb button:last-of-type');
    const crumb = cur ? cur.textContent.trim() : '';
    const det = document.getElementById('lensDetail'), detail = !!(det && !det.hidden);
    const canvas = document.getElementById('viewKernel');
    const focus = level === 1 ? 'graph' : (canvas && canvas.dataset.focus) || 'graph';
    const key = level + '|' + crumb + '|' + detail + '|' + focus;
    if (key === last) return;
    last = key;
    try { window.parent.postMessage({ type: 'lens:level', level: level, crumb: crumb, detail: detail, focus: focus }, '*'); } catch (err) { /* 没有外层时忽略 */ }
  }
  function sourceGeometry(type) {
    const box = document.getElementById('srcFile');
    if (!box) return null;
    const r = box.getBoundingClientRect();
    // Agent 的 qk_pv 和 L1 的 decode_csa 来自不同文件，行号不能当作同一锚点。
    return { type: type, left: r.left, top: r.top, width: r.width, height: r.height, anchorX: null, anchorY: null };
  }
  function reportSourceBounds() {
    const data = sourceGeometry('lens:source-bounds');
    if (data) window.parent.postMessage(data, '*');
  }
  // 嵌入样式、字体和布局生效后再测量目标区域。
  const fontsReady = document.fonts ? Promise.all([document.fonts.ready, document.fonts.load('12px "JetBrains Mono"')]) : Promise.resolve();
  const sourceReady = fontsReady.catch(() => {}).then(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))).then(() => {
    const params = new URLSearchParams(location.search);
    const line = Number(params.get('line'));
    const box = document.getElementById('srcFile');
    const row = box && Number.isInteger(line) && line > 0 && box.querySelector('.cl[data-line="' + line + '"]');
    if (!row) return;
    if (window.workbenchCancelSourceScroll) window.workbenchCancelSourceScroll(box);
    box.scrollTop += row.getBoundingClientRect().top - box.getBoundingClientRect().top - (Number(params.get('offset')) || 0);
    box.scrollLeft = Math.max(0, Number(params.get('scrollX')) || 0);
  });
  window.addEventListener('resize', () => sourceReady.then(reportSourceBounds));
  new MutationObserver(report).observe(document.body, { attributes: true, attributeFilter: ['data-level'] });
  const canvas = document.getElementById('viewKernel');
  if (canvas) new MutationObserver(report).observe(canvas, { attributes: true, attributeFilter: ['data-focus'] });
  const crumb = document.getElementById('lensCrumb');
  if (crumb) new MutationObserver(report).observe(crumb, { childList: true, subtree: true, characterData: true });
  const det = document.getElementById('lensDetail');
  if (det) new MutationObserver(report).observe(det, { attributes: true, attributeFilter: ['hidden'] });

  let agentSource = null, originalMeta = '';
  function clearAgentSource() {
    if (!agentSource) return;
    agentSource.remove(); agentSource = null;
    document.getElementById('srcMeta').textContent = originalMeta;
    window.parent.postMessage({ type: 'lens:agent-source-dismissed' }, '*');
  }
  function positionAgentSource() {
    if (!agentSource) return;
    const box = document.getElementById('srcFile');
    agentSource.style.top = box.offsetTop + 'px';
    agentSource.style.height = box.offsetHeight + 'px';
    agentSource.style.boxSizing = 'border-box';
  }
  window.addEventListener('resize', positionAgentSource);
  const mid = document.querySelector('.hp-mid');
  if (mid) mid.addEventListener('pointerdown', clearAgentSource, true);
  window.addEventListener('message', function (e) {
    const graph = document.getElementById('layerFrame');
    if (graph && e.source === graph.contentWindow && e.data && e.data.type === 'lens:user-interaction') clearAgentSource();
  });

  window.addEventListener('message', function (e) {
    const d = e.data || {};
    if (d.type === 'lens:go') {
      clearAgentSource();
      const b = document.querySelector('#lensLevels button[data-level="' + d.level + '"]');
      if (b) b.click();
    } else if (d.type === 'lens:detail') {
      const t = document.getElementById('lensDetailToggle');
      if (t) t.click();
    } else if (d.type === 'lens:hello') {
      last = '';
      report();
      sourceReady.then(reportSourceBounds);
    } else if (d.type === 'lens:agent-source' && e.source === window.parent && typeof d.html === 'string') {
      clearAgentSource();
      const box = document.getElementById('srcFile');
      agentSource = document.createElement('div');
      agentSource.className = 'agent-source-preview';
      agentSource.setAttribute('aria-label', 'Agent 生成的 qk_pv 源码');
      agentSource.tabIndex = 0;
      agentSource.style.visibility = 'hidden';
      agentSource.innerHTML = d.html;
      agentSource.querySelectorAll('.is-new').forEach(row => row.classList.remove('is-new'));
      box.parentNode.appendChild(agentSource);
      originalMeta = document.getElementById('srcMeta').textContent;
      document.getElementById('srcMeta').textContent = 'decode_sparse_attn_csa.py · qk_pv 198–313';
      positionAgentSource();
      agentSource.scrollTop = Math.max(0, Number(d.scrollTop) || 0);
      agentSource.scrollLeft = Math.max(0, Number(d.scrollLeft) || 0);
      const sourceView = agentSource;
      sourceView.addEventListener('scroll', () => {
        if (agentSource !== sourceView) return;
        window.parent.postMessage({ type: 'lens:agent-source-scroll', scrollTop: sourceView.scrollTop, scrollLeft: sourceView.scrollLeft }, '*');
      }, { passive: true });
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const data = sourceGeometry('lens:agent-source-ready');
        if (data) window.parent.postMessage(data, '*');
      }));
    } else if (d.type === 'lens:show-source') {
      document.body.classList.remove('source-in-flight');
      if (agentSource) agentSource.style.visibility = 'visible';
      else {
        document.body.classList.add('source-reveal');
        const source = document.getElementById('srcFile');
        if (source) source.addEventListener('animationend', () => document.body.classList.remove('source-reveal'), { once: true });
      }
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const data = sourceGeometry('lens:source-shown');
        if (data) window.parent.postMessage(data, '*');
      }));
    }
  });
  report();
  setTimeout(report, 300);
})();
