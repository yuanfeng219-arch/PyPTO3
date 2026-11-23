(() => {
  // 每页用一句总结和一行辅助说明表达核心信息。
  const highlights = {
    'agent.html': [
      '从手写代码转向引导 Agent 生成',
      '写出算子变得简单，但理解和判断依然困难',
      { detail: '理解和判断' }
    ],
    'lens.html': [
      '计算图 · 算什么',
      '将复杂源码可视化，快速看懂从整层到 Kernel 的计算逻辑。'
    ],
    'workbench.html': [
      '上板前看清代码如何执行',
      '对照源码、计算图与芯片落点，检查任务分工和存储分配'
    ],
    'compile.html': [
      '看清编译过程，定位异常',
      '对照图形、代码与诊断信息，追踪各 Pass 的变换'
    ],
    'loading.html': [
      '通过泳道图看清算子的设备性能表现',
      '展示各核任务、耗时与并行情况，辅助分析执行效率。',
      { challenge: '瓶颈发现仍依赖开发者的读图经验。' }
    ],
    'result.html': [
      '抓主线 · 看清关键路径',
      '突出影响整体耗时的任务，帮助开发者聚焦分析主线。'
    ],
    'all-passes.html': [
      '回看完整编译链',
      '按顺序对照各 Pass 的中间状态，理解每一步变换'
    ]
  };

  const page = location.pathname.split('/').pop();
  const embed = new URLSearchParams(location.search).get('embed');
  const embeddedLens = page === 'workbench.html' && embed === 'lens';
  const embeddedResult = page === 'result.html' && embed === 'agent' && window.parent !== window;
  if (embeddedLens || embeddedResult || !highlights[page]) return;

  if (!document.querySelector('link[data-demo-highlights-style]')) {
    const stylesheet = document.createElement('link');
    stylesheet.rel = 'stylesheet';
    stylesheet.href = new URL('demo-highlights.css', document.currentScript.src);
    document.head.append(stylesheet);
  }

  function createPanel(page) {
    const panel = document.createElement('details');
    panel.className = 'demo-highlights';
    panel.open = true;
    panel.dataset.page = page.replace('.html', '');
    const panelTitles = {
      'agent.html': '算子开发的挑战',
      'lens.html': '算子理解三张图',
      'loading.html': '性能分析图',
      'result.html': '泳道排查三步走'
    };
    const summary = document.createElement('summary');
    const title = panelTitles[page] ?? '体验设计亮点';
    summary.innerHTML = `
      <span class="demo-highlights-label">${title}<small>DESIGN NOTES</small></span>
      <svg class="demo-highlights-chevron" width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="m4 6 4 4 4-4"/></svg>`;

    const content = document.createElement('div');
    content.className = 'highlight-brief';
    const headline = document.createElement('p');
    headline.className = 'highlight-brief-main';
    headline.textContent = highlights[page][0];
    const detail = document.createElement('p');
    detail.className = 'highlight-brief-detail';
    const detailText = highlights[page][1];
    const emphasis = highlights[page][2]?.detail;
    const index = emphasis ? detailText.indexOf(emphasis) : -1;
    if (index >= 0) {
      const warning = document.createElement('span');
      warning.className = 'highlight-brief-caution';
      warning.textContent = emphasis;
      detail.append(detailText.slice(0, index), warning, detailText.slice(index + emphasis.length));
    } else {
      detail.textContent = detailText;
    }
    content.append(headline, detail);
    if (highlights[page][2]?.challenge) {
      const challenge = document.createElement('p');
      challenge.className = 'highlight-brief-detail highlight-brief-caution';
      challenge.textContent = highlights[page][2].challenge;
      content.append(challenge);
    }
    panel.append(summary, content);
    return panel;
  }

  const panel = createPanel(page);
  // 触发时机待演示路径确定后接入：发送 step 为 path、focus 或 evidence。
  const analysisSteps = {
    path: ['抓主线 · 看清关键路径', '突出影响整体耗时的任务，帮助开发者聚焦分析主线。'],
    focus: ['看重点 · 缩小排查范围', '突出低并行度等线索，帮助开发者发现值得关注的任务。'],
    evidence: ['追证据 · 理解问题原因', '关联任务、核内数据与关键 Pass，帮助开发者分析问题原因。']
  };
  function showAnalysisStep(target, step) {
    const copy = analysisSteps[step];
    if (!target || !copy) return;
    target.querySelector('.highlight-brief-main').textContent = copy[0];
    target.querySelector('.highlight-brief-detail').textContent = copy[1];
  }
  if (page === 'result.html' || page === 'loading.html') {
    window.addEventListener('demo:analysis-step', event => {
      const target = page === 'result.html' ? panel : document.querySelector('.analysis-shell.is-result .demo-highlights[data-page="result"]');
      showAnalysisStep(target, event.detail?.step);
    });
    if (page === 'loading.html') window.addEventListener('message', event => {
      if (event.data?.type !== 'demo:analysis-step') return;
      const frame = document.querySelector('.analysis-shell.is-result iframe');
      if (frame && event.source === frame.contentWindow) {
        showAnalysisStep(frame.closest('.analysis-shell').querySelector('.demo-highlights[data-page="result"]'), event.data.step);
      }
    });
  }
  let lensView = 'graph';
  function updateLensPanel(target, state) {
    if (state.level === 1 || state.focus === 'graph') lensView = 'graph';
    else if (state.focus === 'work') lensView = 'work';
    else if (state.focus === 'chip' || state.focus === 'linked') lensView = 'chip';
    if (!target) return;
    const copy = {
      graph: ['计算图 · 算什么', '将复杂源码可视化，快速看懂从整层到 Kernel 的计算逻辑。'],
      work: ['任务分配 · 数据怎么分', '看清数据切分与 SPMD Lane 分工，快速理解任务如何分配。'],
      chip: ['存储落点 · 落在哪儿算', '关联源码与昇腾硬件，直观看清 AIC / AIV 的存储落点。']
    }[lensView];
    target.querySelector('.highlight-brief-main').textContent = copy[0];
    target.querySelector('.highlight-brief-detail').textContent = copy[1];
  }
  if (page === 'agent.html') {
    // file:// 下父子页面可能不能直接访问 DOM，使用消息让外层创建 Lens 说明。
    window.addEventListener('message', event => {
      const frame = document.querySelector('.ag-fly > iframe');
      if (!frame || event.source !== frame.contentWindow) return;
      if (event.data?.type === 'lens:level') {
        updateLensPanel(document.querySelector('.demo-highlights.is-hosted-lens'), event.data);
        return;
      }
      if (event.data?.type !== 'highlights:host-lens') return;
      document.querySelector('.demo-highlights.is-hosted-lens')?.remove();
      const hosted = createPanel('lens.html');
      hosted.classList.add('is-hosted-lens');
      document.body.append(hosted);
      updateLensPanel(hosted, { level: 2, focus: lensView });
      event.source.postMessage({ type: 'highlights:lens-hosted' }, '*');
    });
  }
  if (page === 'lens.html') {
    window.addEventListener('message', event => {
      const frame = document.getElementById('view');
      if (frame && event.source === frame.contentWindow && event.data?.type === 'lens:level') updateLensPanel(panel, event.data);
    });
  }
  if (page === 'lens.html' && new URLSearchParams(location.search).has('transition') && window.parent !== window) {
    let hosted = false;
    window.addEventListener('message', event => {
      if (event.source !== window.parent || event.data?.type !== 'highlights:lens-hosted') return;
      hosted = true;
      panel.remove();
    });
    window.parent.postMessage({ type: 'highlights:host-lens' }, '*');
    // 非 Agent 宿主仍可在 iframe 内展示说明；收到确认后会移除备用面板。
    setTimeout(() => { if (!hosted) document.body.append(panel); }, 500);
  } else {
    document.body.append(panel);
  }
  if (page === 'loading.html') {
    // 卡片随 Agent 工作台挂载，定位使用外层视口。
    document.getElementById('analysisShell').content.firstElementChild.append(createPanel('result.html'));
  }
})();
