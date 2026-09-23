/* ---- 统一算子属性面板渲染器 ------------------------------------------------
   一个 renderOperatorInspector(profile) 服务所有算子源码。四层骨架：

     ① 结论区   一句话语义 · 身份行 · 状态条（风险 chips ｜ 置信度 chips）
   ② Tab 条   恒定四格：概览 / 数据&精度 / 分块硬件 / 编排依赖
     ③ Body     标准区块拼装
     ④ 证据与行动  五级来源图例 + 主行动

   风险与置信度**不出现在任何 tab 内** —— 它们在结论区常驻，点开共用抽屉。
   新增一个算子 = 新增一个 OperatorProfile，不改这个文件。

   规范：算子属性面板统一规范 §四–§九 */
(function () {
  'use strict';

  const TABS = [
    ['overview', '概览'],
    ['data', '数据&精度'],
    ['tiling', '分块硬件'],
    ['orch', '编排依赖'],
  ];

  // 五级来源（规划文档 §3.2 口径）。后两级指**已经取到的证据**，不是"以后可确认"。
  const ORIGINS = {
    fact: '源码事实',
    resolved: '跨文件解析',
    compiled: '编译事实',
    estimated: '静态估算',
    measured: '运行实测',
  };

  const DEPTH_LABEL = {
    fact: 'AST 解析',
    resolved: '跨文件解析',
    compiled: '编译事实',
    estimated: '静态估算',
    measured: '运行实测',
  };

  const COMPLETENESS = {
    complete: '完整实现',
    partial: '部分实现',
    placeholder: '占位',
    blocked: '编译受阻',
    unmodeled: '未建模',
  };

  // 置信度五项——把"编译通过"拆成一份声明（洞察报告 §4.5）。
  // chip 用短名保证状态条恒为一行，抽屉里给全称与具体证据。
  const CONFIDENCE = [
    ['compiled', '编译', '编译', '程序可通过 ir.compile()'],
    ['torchGolden', '数值', 'Torch 数值比对', '有 Torch Golden 并执行数值比对'],
    ['passDump', 'Pass', '逐 Pass 校验', '对每个 passes_dump 快照逐一数值校验'],
    ['device', '设备', '设备实跑', '在昇腾设备上实跑并校验输出'],
    ['perfBaseline', '性能', '性能基线', '有可比的吞吐 / 时延基线'],
  ];

  const esc = (value) => String(value == null ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  function originTag(origin) {
    if (!origin || !ORIGINS[origin]) return '';
    return `<span class="kf-op__origin"><i class="kf-op-dot-${origin}"></i>${ORIGINS[origin]}</span>`;
  }

  // Footer 图例用短标签、title 带完整口径；区块头的 originTag 仍逐字用 ORIGINS 全称，
  // 所以"来源文案全局唯一"这条仍然成立——短标签只是同一套词的索引。
  const LEGEND_SHORT = { fact: '源码', resolved: '跨文件', compiled: '编译', estimated: '估算', measured: '实测' };

  function legend() {
    return Object.entries(ORIGINS)
      .map(([key, label]) => `<span title="${label}"><i class="kf-op-dot-${key}"></i>${LEGEND_SHORT[key]}</span>`)
      .join('');
  }

  /* ---- 标准区块 -------------------------------------------------------- */

  function renderBlock(block) {
    if (!block) return '';
    const html = typeof block.html === 'function' ? block.html() : block.html;
    switch (block.type) {
      // 逃生口：沿用现有已成型的 section（它们自带 header）。
      // 每迁移一个文件就把其中一部分抽成结构化区块，这个分支会持续收缩。
      case 'raw':
        return html || '';
      case 'na':
        return `<div class="kf-op__na">
          <b>本格对当前入口不适用</b>
          <p>${esc(block.reason)}</p>
          ${block.jumpTo ? `<button type="button" data-op-jump="${esc(block.jumpTo)}">${esc(block.jumpLabel || block.jumpTo)} →</button>` : ''}
        </div>`;
      case 'note':
        return `<div class="kf-op__empty-note"><b>${esc(block.title)}</b>${html || esc(block.text)}</div>`;
      default:
        return `<section class="kf-op__block">
          ${block.title ? `<header class="kf-op__block-head"><h3>${esc(block.title)}</h3>${originTag(block.origin)}</header>` : ''}
          ${html || ''}
        </section>`;
    }
  }

  /* ---- 抽屉内容 -------------------------------------------------------- */

  const LEVEL_LABEL = { block: '阻塞', warn: '警告', guard: '护栏' };

  function riskItem(item) {
    const lines = Array.isArray(item.lines) ? item.lines : null;
    const lineLabel = lines ? (lines[0] === lines[1] ? `第 ${lines[0]} 行` : `第 ${lines[0]}–${lines[1]} 行`) : null;
    // 整个标题行都是跳转源码的按钮，不只是右边那枚小行号——风险列表最常见的
    // 用法是逐条点下去看代码，点击目标越大越好。正文留给选中复制，不参与点击。
    return `<article class="kf-op__risk-item is-${esc(item.level)}"${lines ? ` data-op-risk-line="${lines[0]}"` : ''}>
      <div class="kf-op__risk-top${lines ? ' is-clickable' : ''}"${lines ? ` data-op-goto-line="${lines[0]}" role="button" tabindex="0"` : ''}>
        <b>${esc(item.title)}</b>
        <span class="kf-op__risk-cls">${esc(item.cls)} · ${esc(LEVEL_LABEL[item.level] || item.level)}</span>
        ${lines ? `<span class="kf-op__risk-line">${esc(lineLabel)}</span>` : ''}
      </div>
      <dl>
        <div><dt>为什么</dt><dd>${item.why}</dd></div>
        <div><dt>影响什么</dt><dd>${item.impact}</dd></div>
        ${item.fix ? `<div class="is-fix"><dt>建议改法</dt><dd>${item.fix}</dd></div>` : ''}
        <div><dt>如何验证</dt><dd>${item.verify}</dd></div>
      </dl>
      ${item.conflict ? `<p class="kf-op__risk-conflict"><b>与现有面板结论冲突：</b>${item.conflict}</p>` : ''}
    </article>`;
  }

  function guardItem(item) {
    const lines = Array.isArray(item.lines) ? item.lines : null;
    return `<article class="kf-op__risk-item is-guard">
      <div class="kf-op__risk-top">
        <b>${esc(item.api)}</b>
        <span class="kf-op__risk-cls">带外承诺</span>
        ${lines ? `<button type="button" class="kf-op__risk-line" data-op-goto-line="${lines[0]}">第 ${lines[0]} 行</button>` : ''}
      </div>
      <dl>
        <div><dt>承诺了什么</dt><dd>${item.promise}</dd></div>
        <div><dt>谁在依赖</dt><dd>${item.dependents}</dd></div>
        <div><dt>删掉会怎样</dt><dd>${item.ifRemoved}</dd></div>
      </dl>
    </article>`;
  }

  function drawerContent(profile, drawer) {
    if (drawer === 'guards') {
      const guards = profile.guards || [];
      if (!guards.length) {
        return {
          title: '语义护栏',
          meta: '0 处带外承诺',
          html: `<div class="kf-op__empty-note">
            <b>本文件没有语义护栏</b>
            源码全程运行在 AUTO scope 下，没有出现 <code>pl.no_dep</code>、<code>manual_dep=True</code>、
            <code>no_dep_args=</code> 或 <code>pl.manual_scope()</code>；任务顺序完全由 Tensor 的生产 / 消费
            与 <code>InOut</code> 状态自动推导。<br><br>
            这是一条<b>有价值的空结论</b>：读这段代码时不存在"删不得但说不清为什么"的行，
            所有语句都可以按字面意思理解。
          </div>`,
        };
      }
      return { title: '语义护栏', meta: `${guards.length} 处带外承诺`, html: `<div class="kf-op__risk-list">${guards.map(guardItem).join('')}</div>` };
    }

    if (drawer === 'block' || drawer === 'warn') {
      const level = drawer;
      const items = (profile.risks || []).filter((item) => item.level === level);
      const title = level === 'block' ? '阻塞风险' : '警告风险';
      if (!items.length) {
        return {
          title,
          meta: '0 项',
          html: `<div class="kf-op__empty-note"><b>没有${title}</b>静态检查未在源码里发现该级别的问题。这不代表运行期一定正确——右下角的证据图例说明了每块内容的可信来源。</div>`,
        };
      }
      return { title, meta: `${items.length} 项 · 每项含为什么 / 影响什么 / 如何验证`, html: `<div class="kf-op__risk-list">${items.map(riskItem).join('')}</div>` };
    }

    if (drawer === 'confidence') {
      const conf = profile.confidence || {};
      return {
        title: '置信度',
        meta: '"编译通过"拆成五项',
        html: `<div class="kf-op__evidence">${CONFIDENCE.map(([key, , label, what]) => {
          const entry = conf[key];
          const on = entry === true || (entry && entry.ok);
          const note = entry && entry.note ? entry.note : what;
          return `<div class="${on ? 'is-on' : ''}"><i>${on ? '✓' : '✗'}</i><span><b>${esc(label)}</b><small>${note}</small></span><em>${on ? '有证据' : '缺失'}</em></div>`;
        }).join('')}</div>${profile.confidenceNote ? `<div class="kf-op__evidence-note">${typeof profile.confidenceNote === 'function' ? profile.confidenceNote() : profile.confidenceNote}</div>` : ''}`,
      };
    }

    if (drawer && profile.drawers && profile.drawers[drawer]) {
      const custom = profile.drawers[drawer];
      return { title: custom.title, meta: custom.meta, html: typeof custom.html === 'function' ? custom.html() : custom.html };
    }
    return null;
  }

  /* ---- 主渲染 ---------------------------------------------------------- */

  function render(options) {
    const { mount, titleEl, metaEl, profile } = options;
    const tab = TABS.some(([key]) => key === options.tab) ? options.tab : 'overview';
    const drawer = options.drawer || null;
    const mode = options.mode === 'diff' ? 'diff' : 'read';
    if (!mount || !profile) return;

    const risks = profile.risks || [];
    const guards = profile.guards || [];
    const blockCount = risks.filter((item) => item.level === 'block').length;
    const warnCount = risks.filter((item) => item.level === 'warn').length;
    const guardCount = guards.length;
    const clean = !blockCount && !warnCount && !guardCount;

    if (titleEl) titleEl.textContent = '算子分析';
    // 文件路径不写进头部：正上方的编辑器标签页已经在显示同一个路径，头部再写
    // 一遍只是把标题挤窄、逼出省略号。profile.fileLabel 仍然保留，供别处引用。
    if (metaEl) metaEl.textContent = '';

    const badges = [
      `<span class="kf-op__badge is-kind">${esc(profile.kindLabel || profile.kind)}</span>`,
      profile.completeness ? `<span class="kf-op__badge is-${esc(profile.completeness)}">${esc(COMPLETENESS[profile.completeness] || profile.completeness)}</span>` : '',
      ...(profile.traits || []).map((trait) => `<span class="kf-op__badge is-trait">${esc(trait)}</span>`),
    ].join('');

    // 状态条左组：位置恒定占用，计数为 0 时也不塌陷——状态才可读。
    const riskChips = clean
      ? `<button type="button" class="is-clear" disabled>静态检查通过 · ${profile.checkCount || 0} 项</button>`
      : `<button type="button" class="is-block" data-op-drawer="block"${blockCount ? '' : ' disabled'}>⛔ ${blockCount}</button>
         <button type="button" class="is-warn" data-op-drawer="warn"${warnCount ? '' : ' disabled'}>⚠ ${warnCount}</button>
         <button type="button" class="is-guard" data-op-drawer="guards">🛡 ${guardCount}</button>`;

    // 多函数文件的作用域切换器。一个文件里几个同族函数（rmsnorm 的 input/post）
    // 共用一套结论与风险，但数据与分块各不相同——切换它，Body 换内容，骨架不动。
    const variants = profile.variants || [];
    const activeVariant = variants.some((v) => v.id === options.variant) ? options.variant : (variants[0] && variants[0].id);
    const variantSwitch = variants.length > 1
      ? `<div class="kf-op__variants" role="group" aria-label="${esc(profile.variantLabel || '函数')}">${variants.map((v) => `<button type="button" class="${v.id === activeVariant ? 'is-active' : ''}" data-op-variant="${esc(v.id)}"><b>${esc(v.name)}</b><small>${esc(v.role || '')}</small></button>`).join('')}</div>`
      : '';

    const conf = profile.confidence || {};
    const confChips = CONFIDENCE.map(([key, shortLabel, fullLabel]) => {
      const entry = conf[key];
      const on = entry === true || (entry && entry.ok);
      return `<button type="button" class="${on ? 'is-on' : 'is-off'}" data-op-drawer="confidence" title="${esc(fullLabel)}"><i>${on ? '✓' : '✗'}</i>${esc(shortLabel)}</button>`;
    }).join('');

    const tabsHtml = TABS.map(([key, label]) => {
      const blocks = (profile.tabs && profile.tabs[key]) || [];
      const isNa = blocks.length === 1 && blocks[0].type === 'na';
      return `<button type="button" role="tab" aria-selected="${key === tab}" class="${key === tab ? 'is-active' : ''}${isNa ? ' is-empty' : ''}" data-op-tab="${key}">${label}</button>`;
    }).join('');

    const bodyHtml = ((profile.tabs && profile.tabs[tab]) || []).map(renderBlock).join('');

    const action = (profile.actions && profile.actions[mode]) || null;
    const drawerData = drawer ? drawerContent(profile, drawer) : null;

    mount.classList.add('is-op-inspector');
    mount.innerHTML = `<div class="kf-op${mode === 'diff' ? ' is-diff' : ''}">
      <header class="kf-op__verdict">
        <div class="kf-op__eyebrow">SOURCE ANALYSIS · ${esc(DEPTH_LABEL[profile.depth] || profile.depth || '')}</div>
        <p class="kf-op__summary" data-op-summary>${profile.summary || ''}</p>
        <div class="kf-op__id"><b class="kf-op__name">${esc(profile.name)}</b>${badges}</div>
        ${variantSwitch}
        <div class="kf-op__status">
          <div class="kf-op__risk">${riskChips}</div>
          <div class="kf-op__status-div"></div>
          <div class="kf-op__confidence">${confChips}</div>
        </div>
      </header>

      <div class="kf-op__tabs" role="tablist" aria-label="算子分析视图">${tabsHtml}</div>

      <div class="kf-op__body" data-op-body>${bodyHtml}</div>

      <footer class="kf-op__foot">
        <div class="kf-op__legend">${legend()}</div>
        ${action ? `<button type="button" class="kf-op__action" data-op-action="${esc(action.id)}">${esc(action.label)}</button>` : ''}
      </footer>

      ${drawerData ? `
        <button type="button" class="kf-op__scrim" data-op-drawer-close aria-label="关闭抽屉"></button>
        <aside class="kf-op__drawer" role="dialog" aria-label="${esc(drawerData.title)}">
          <header class="kf-op__drawer-head">
            <b>${esc(drawerData.title)}</b><small>${esc(drawerData.meta || '')}</small>
            <button type="button" class="kf-op__drawer-close" data-op-drawer-close aria-label="关闭">✕</button>
          </header>
          <div class="kf-op__drawer-body">${drawerData.html}</div>
        </aside>` : ''}
    </div>`;
  }

  /* ---- 源码 ↔ 面板的唯一锚点 -------------------------------------------
     行号到阶段的映射只在 profile.scopes[].lines 声明一次。此前每个文件各有
     一条硬编码 if 链（`lineNumber < 93 ? 'qk' : ...`），源码一改就默默错位。 */

  function scopeForLine(profile, line) {
    const scopes = (profile && profile.scopes) || [];
    for (const scope of scopes) {
      const range = scope.lines;
      if (Array.isArray(range) && line >= range[0] && line <= range[1]) return scope.id;
    }
    return null;
  }

  function scopeById(profile, id) {
    return ((profile && profile.scopes) || []).find((scope) => scope.id === id) || null;
  }

  // 显示用行号：默认是归属区间，`anchor` 可以收窄到真正有内容的几行
  function scopeLines(scope) {
    if (!scope) return '—';
    if (scope.anchor) return scope.anchor;
    const range = scope.lines || [];
    return range[0] === range[1] ? String(range[0]) : `${range[0]}–${range[1]}`;
  }

  window.PtoOperatorInspector = { render, TABS, ORIGINS, CONFIDENCE, scopeForLine, scopeById, scopeLines };
})();
