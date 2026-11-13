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

  const css = document.createElement('style');
  css.textContent =
    'body.is-embedded .hp-top { display:none !important; }\n' +
    'body.is-embedded .hp-app { grid-template-rows:minmax(0,1fr) 28px; }\n' +
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
    const key = level + '|' + crumb + '|' + detail;
    if (key === last) return;
    last = key;
    try { window.parent.postMessage({ type: 'lens:level', level: level, crumb: crumb, detail: detail }, '*'); } catch (err) { /* 没有外层时忽略 */ }
  }
  new MutationObserver(report).observe(document.body, { attributes: true, attributeFilter: ['data-level'] });
  const crumb = document.getElementById('lensCrumb');
  if (crumb) new MutationObserver(report).observe(crumb, { childList: true, subtree: true, characterData: true });
  const det = document.getElementById('lensDetail');
  if (det) new MutationObserver(report).observe(det, { attributes: true, attributeFilter: ['hidden'] });

  window.addEventListener('message', function (e) {
    const d = e.data || {};
    if (d.type === 'lens:go') {
      const b = document.querySelector('#lensLevels button[data-level="' + d.level + '"]');
      if (b) b.click();
    } else if (d.type === 'lens:detail') {
      const t = document.getElementById('lensDetailToggle');
      if (t) t.click();
    } else if (d.type === 'lens:hello') {
      last = '';
      report();
    }
  });
  report();
  setTimeout(report, 300);
})();
