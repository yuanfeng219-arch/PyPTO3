(() => {
  'use strict';

  const SUGGESTION_META = {
    'assign-balance': {
      short: '重排 Assignment 分核映射',
      full: '建议 1 · 重排 Assignment 分核映射（打散 24 核组同余碰撞）',
      prompt: '我在 Assignment 视角确认了 12 慢 / 12 快核组分核碰撞问题，请按建议 1 修改代码，把高代价压缩块均匀分配到 24 个核组（保持计算图拓扑不变）。改完后把新算子的决策路径和静态三视角发我核对。',
    },
    'qk-plan-spmd': {
      short: '拆分 qk_plan 单核串行屏障',
      full: '建议 2 · 拆分 qk_plan 单核串行屏障（pl.spmd 并行化）',
      prompt: '请按建议 2 修改代码，拆分 qk_plan 单核串行屏障并并行执行，改完后给我看新算子的决策路径和静态三视角。',
    },
    'skip-invalid': {
      short: '透传 valid_block_mask 跳过无效块零写',
      full: '建议 3 · 透传 valid_block_mask 跳过无效块零写',
      prompt: '请按建议 3 修改代码，将 valid_block_mask 透传至下游 Merge 并跳过无效块零写，改完后给我看新算子的决策路径和静态三视角。',
    },
  };

  const VIEWS = {
    decision: {
      id: 'decision',
      cardKey: 'decision',
      badge: 'v1 · 开发决策',
      title: 'Agent 决策路径 · Decision Explorer (v1)',
      url: '../agent-decision-lineage/agent-decision-lineage-softmax.html',
      note: '展示初版算子（v1）开发时的五阶段决策路径、设计取舍与实验依据；左侧对话流可随时上下滚动。',
    },
    static: {
      id: 'static',
      cardKey: 'static',
      badge: 'v1 · 静态分析',
      title: '静态三视角分析 · Graph / Pipeline / Assignment (v1)',
      url: '../PyPTO-Compute-Explorer-UI/index.html',
      note: '上板前静态分析视角（v1）：可在顶部切换 Graph、Pipeline、Assignment 三个视图。',
    },
    lane: {
      id: 'lane',
      cardKey: 'lane',
      badge: '上板诊断 · 全量泳道',
      title: '全量运行泳道图 · Device Perf Lane',
      url: '../PyPTO-Compute-Explorer-UI/perf-lane-slz/result.html',
      note: '全量 72 核执行泳道图：点击顶部异常结论中的“深入调查”，直接下钻对比 Runtime Trace 与静态 Pipeline。',
    },
    'csa-pipeline': {
      id: 'csa-pipeline',
      cardKey: 'lane',
      badge: '深入调查 · Pipeline vs Trace',
      title: 'Runtime Trace vs 静态 Pipeline · Decode CSA',
      url: '../PyPTO-Compute-Explorer-UI/csa.html',
      csaTab: 'pipeline',
      note: '正在对比实测 Runtime Trace 与静态 Pipeline：观察 12 慢 / 12 快核组分化，切换至 Assignment 查看根因与内嵌修改建议。',
    },
    'csa-assignment': {
      id: 'csa-assignment',
      cardKey: 'lane',
      badge: '根因定位 · Assignment',
      title: 'Assignment 分核根因与修改建议 · Decode CSA',
      url: '../PyPTO-Compute-Explorer-UI/csa.html',
      csaTab: 'assignment',
      note: '已定位 Assignment 分核根因：右下角浮窗与顶栏均提供修改意见与“修改代码”入口，点击后直接进入下一轮修改交付。',
    },
    'decision-v2': {
      id: 'decision-v2',
      cardKey: 'decision-v2',
      badge: 'v2 · 新算子决策路径',
      title: '新算子决策路径 · Decision Explorer (v2)',
      url: '../agent-decision-lineage/agent-decision-lineage-softmax.html',
      note: 'v2 修改完成后的新算子决策路径：聚焦 Assignment 分核均衡重构依据与回归校验结果。',
    },
    'static-v2': {
      id: 'static-v2',
      cardKey: 'static-v2',
      badge: 'v2 · 新算子静态分析',
      title: '新算子静态三视角 · Graph / Pipeline / Assignment (v2)',
      url: '../PyPTO-Compute-Explorer-UI/index.html',
      note: 'v2 修改完成后的新算子静态三视角：核对计算图拓扑不变、Assignment 分核均衡且流水长尾收敛。',
    },
  };

  const shell = document.getElementById('appShell');
  const stagePane = document.getElementById('stagePane');
  const stageFrame = document.getElementById('stageFrame');
  const stageLoading = document.getElementById('stageLoading');
  const stageBadge = document.getElementById('stageBadge');
  const stageTitle = document.getElementById('stageTitle');
  const stageNote = document.getElementById('stageNote');
  const stageContextActions = document.getElementById('stageContextActions');
  const backToLaneBtn = document.getElementById('backToLaneBtn');
  const openRawLink = document.getElementById('openRawLink');
  const toggleChatSplitBtn = document.getElementById('toggleChatSplitBtn');
  const toggleCanvasBtn = document.getElementById('toggleCanvasBtn');
  const closeStageBtn = document.getElementById('closeStageBtn');
  const sidebarCollapseBtn = document.getElementById('sidebarCollapseBtn');
  const sidebarOpenBtn = document.getElementById('sidebarOpenBtn');
  const newThreadBtn = document.getElementById('newThreadBtn');
  const chatScroll = document.getElementById('chatScroll');
  const chatThread = document.getElementById('chatThread');
  const composerForm = document.getElementById('composerForm');
  const composerInput = document.getElementById('composerInput');
  const toastEl = document.getElementById('toast');

  const suggestionsSummary = document.getElementById('suggestionsSummary');
  const applyChatSuggestionsBtn = document.getElementById('applyChatSuggestionsBtn');
  const appliedPlanSource = document.getElementById('appliedPlanSource');
  const appliedPlanText = document.getElementById('appliedPlanText');
  const refactorUserPrompt = document.getElementById('refactorUserPrompt');
  const turnRefactor = document.getElementById('turn-refactor');
  const turnRefactorResult = document.getElementById('turn-refactor-result');

  let activeViewId = null;
  let loadedUrl = '';
  let cameFromLane = false;
  let currentCsaTab = 'pipeline';
  let bridgeCleanup = null;
  let toastTimer = 0;

  function showToast(text) {
    clearTimeout(toastTimer);
    toastEl.textContent = text;
    toastEl.hidden = false;
    toastTimer = setTimeout(() => {
      toastEl.hidden = true;
    }, 3000);
  }

  function setSidebar(open) {
    shell.dataset.sidebar = open ? 'open' : 'collapsed';
  }

  function setLayout(mode) {
    shell.dataset.layout = mode;
    const isStageVisible = mode !== 'chat-only';
    stagePane.hidden = !isStageVisible;
    toggleCanvasBtn.hidden = !activeViewId;
    toggleCanvasBtn.textContent = isStageVisible ? '仅看对话' : '恢复视图画布';
    toggleChatSplitBtn.textContent = mode === 'stage-only' ? '展开对话流' : '收起对话';
  }

  function highlightActiveCard(cardKey) {
    document.querySelectorAll('[data-view-card]').forEach(card => {
      card.classList.toggle('is-active', Boolean(cardKey) && card.dataset.viewCard === cardKey);
    });
  }

  /* 读取当前在对话中勾选的修改建议 */
  function getSelectedSuggestionIds() {
    return [...document.querySelectorAll('input[name="perf-suggestion"]:checked')].map(el => el.value);
  }

  function syncSuggestionsUi() {
    const inputs = [...document.querySelectorAll('input[name="perf-suggestion"]')];
    inputs.forEach(input => {
      input.closest('.suggestion-item')?.classList.toggle('is-selected', input.checked);
    });
    const selected = inputs.filter(i => i.checked).map(i => SUGGESTION_META[i.value]?.short).filter(Boolean);
    if (selected.length === 0) {
      suggestionsSummary.textContent = '请至少勾选 1 项修改建议';
      applyChatSuggestionsBtn.disabled = true;
    } else {
      suggestionsSummary.textContent = `已选 ${selected.length} 项：${selected.join(' + ')}`;
      applyChatSuggestionsBtn.disabled = false;
    }
  }

  /* 统一触发“修改代码”：支持来自 Assignment HTML 内的 CTA，或来自对话流里勾选建议 */
  function triggerModifyCode({ source, suggestionIds }) {
    const ids = suggestionIds?.length ? suggestionIds : ['assign-balance'];
    const labels = ids.map(id => SUGGESTION_META[id]?.full).filter(Boolean);

    // 同步勾选态
    document.querySelectorAll('input[name="perf-suggestion"]').forEach(input => {
      input.checked = ids.includes(input.value);
    });
    syncSuggestionsUi();

    // 更新 Turn 3 的需求来源与内容，让用户立刻看到闭环联动
    appliedPlanSource.textContent =
      source === 'html-cta'
        ? '来自 Assignment 视图「修改代码」CTA'
        : '来自性能诊断 Brief 选中的修改建议';
    appliedPlanText.textContent = labels.join('；');

    if (ids.length === 1 && SUGGESTION_META[ids[0]]) {
      refactorUserPrompt.textContent = SUGGESTION_META[ids[0]].prompt;
    } else {
      const shorts = ids.map(id => SUGGESTION_META[id]?.short).filter(Boolean).join('、');
      refactorUserPrompt.textContent = `请按选中的修改建议执行代码重构：${shorts}。完成后把新算子的决策路径和静态三视角发我核对。`;
    }

    // 回到完整对话或保持分屏并滚动到 Turn 3（修改完成 + 新版视图）
    if (shell.dataset.layout === 'stage-only') {
      setLayout('split');
    } else if (source === 'html-cta') {
      // 从 HTML 内点“修改代码”时，展示对话流并把焦点带到修改完成的新一轮对话
      setLayout('split');
    }

    requestAnimationFrame(() => {
      turnRefactor.scrollIntoView({ behavior: 'smooth', block: 'start' });
      [turnRefactor, turnRefactorResult].forEach(el => {
        el.classList.remove('is-just-focused');
        void el.offsetWidth;
        el.classList.add('is-just-focused');
      });
    });

    showToast('已触发修改代码 · Agent 已按建议完成重构，可在下方查看新算子的决策路径与静态三视角');
  }

  function renderStageChrome(view) {
    stageBadge.textContent = view.badge;
    stageTitle.textContent = view.title;
    openRawLink.href = view.url;
    backToLaneBtn.hidden = !(cameFromLane && (view.id === 'csa-pipeline' || view.id === 'csa-assignment'));

    if (view.note) {
      stageNote.innerHTML = `<span>${view.note}</span>`;
      stageNote.hidden = false;
    } else {
      stageNote.hidden = true;
    }

    stageContextActions.innerHTML = '';

    if (view.id === 'lane') {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn-signal btn-sm';
      btn.textContent = '深入调查异常 · 对比 Runtime Trace 与静态 Pipeline →';
      btn.addEventListener('click', () => {
        openView('csa-pipeline', { fromLane: true });
      });
      stageContextActions.append(btn);
    } else if (view.id === 'csa-pipeline' || view.id === 'csa-assignment') {
      const pipeBtn = document.createElement('button');
      pipeBtn.type = 'button';
      pipeBtn.className = 'btn-outline btn-sm' + (currentCsaTab === 'pipeline' ? ' is-active' : '');
      pipeBtn.textContent = '1. Runtime Trace vs 静态 Pipeline';
      pipeBtn.addEventListener('click', () => switchCsaTab('pipeline'));

      const assignBtn = document.createElement('button');
      assignBtn.type = 'button';
      assignBtn.className = 'btn-outline btn-sm' + (currentCsaTab === 'assignment' ? ' is-active' : '');
      assignBtn.textContent = '2. Assignment 根因';
      assignBtn.addEventListener('click', () => switchCsaTab('assignment'));

      stageContextActions.append(pipeBtn, assignBtn);

      if (currentCsaTab === 'assignment') {
        const modifyBtn = document.createElement('button');
        modifyBtn.type = 'button';
        modifyBtn.className = 'btn-signal btn-sm';
        modifyBtn.textContent = '修改代码 →';
        modifyBtn.addEventListener('click', () => {
          triggerModifyCode({ source: 'html-cta', suggestionIds: ['assign-balance'] });
        });
        stageContextActions.append(modifyBtn);
      }
    }
  }

  function clearBridge() {
    if (typeof bridgeCleanup === 'function') {
      try {
        bridgeCleanup();
      } catch {}
    }
    bridgeCleanup = null;
  }

  function pollUntil(fn, maxMs = 6000, intervalMs = 120) {
    const start = Date.now();
    let timer = 0;
    let stopped = false;
    function tick() {
      if (stopped) return;
      let done = false;
      try {
        done = Boolean(fn());
      } catch {
        done = false;
      }
      if (!done && Date.now() - start < maxMs) {
        timer = setTimeout(tick, intervalMs);
      }
    }
    tick();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }

  /* 桥接 1：决策路径页，收起其内置重复的侧边聊天栏，让外层 Codex 对话成为唯一主线 */
  function bridgeDecisionPage(win, doc) {
    return pollUntil(() => {
      const collapseBtn = doc.querySelector('#agent-chat .chat-header button');
      const reopenBtn = doc.getElementById('chat-reopen');
      if (!collapseBtn) return false;
      const chatPane = doc.getElementById('agent-chat');
      if (chatPane && !chatPane.hidden) {
        collapseBtn.click();
      }
      if (reopenBtn) {
        reopenBtn.style.display = 'none';
      }
      return true;
    }, 5000, 140);
  }

  /* 桥接 2：静态三视角页，若停留在源码页则自动点开 Compute Explorer 三视角 */
  function bridgeStaticExplorer(win, doc) {
    return pollUntil(() => {
      const vizRoot = doc.querySelector('.viz2');
      if (vizRoot && !vizRoot.hidden) return true;
      const openBtn = doc.querySelector('[data-visualization2-open]');
      if (!openBtn) return false;
      openBtn.click();
      return Boolean(doc.querySelector('.viz2:not([hidden])'));
    }, 6000, 160);
  }

  /* 桥接 3：全量泳道图，在顶部首要异常卡片和任务详情里注入“深入调查”按钮，直接带状态跳入 CSA */
  function bridgePerfLane(win, doc) {
    const styleId = 'story-lane-drilldown-style';
    if (!doc.getElementById(styleId)) {
      const style = doc.createElement('style');
      style.id = styleId;
      style.textContent = `
        .story-drilldown-btn {
          margin-left: auto;
          align-self: center;
          flex: none;
          padding: 7px 14px;
          border-radius: 999px;
          border: 1px solid rgba(255, 122, 89, 0.72);
          background: rgba(255, 122, 89, 0.2);
          color: #fff2ee;
          font: 500 12.5px/1.2 Inter, "PingFang SC", sans-serif;
          cursor: pointer;
          white-space: nowrap;
          transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
        }
        .story-drilldown-btn:hover {
          background: rgba(255, 122, 89, 0.36);
          border-color: #ff8f73;
          transform: translateY(-1px);
        }
        .story-drawer-drilldown {
          margin: 10px 0 6px;
          width: 100%;
          padding: 8px 12px;
          border-radius: 8px;
          border: 1px solid rgba(255, 122, 89, 0.65);
          background: rgba(255, 122, 89, 0.16);
          color: #ffe7e0;
          font: 500 12.5px/1.3 Inter, "PingFang SC", sans-serif;
          cursor: pointer;
          text-align: center;
        }
        .story-drawer-drilldown:hover {
          background: rgba(255, 122, 89, 0.28);
        }
      `;
      doc.head.append(style);
    }

    const triggerDrilldown = (ev, contextLabel) => {
      ev.preventDefault();
      ev.stopPropagation();
      if (contextLabel) {
        showToast(`正在深入调查：${contextLabel} → Runtime Trace vs 静态 Pipeline`);
      }
      openView('csa-pipeline', { fromLane: true });
    };

    const inject = () => {
      const lead = doc.getElementById('lead-finding');
      if (lead && !lead.hidden && !lead.querySelector('.story-drilldown-btn')) {
        const title = lead.querySelector('.lead-id b')?.textContent?.trim() || '关键路径异常任务';
        const btn = doc.createElement('button');
        btn.type = 'button';
        btn.className = 'story-drilldown-btn';
        btn.textContent = '深入调查：对比静态 Pipeline →';
        btn.addEventListener('click', ev => triggerDrilldown(ev, title));
        lead.append(btn);
      }

      const drawer = doc.getElementById('drawer');
      if (drawer && !drawer.hidden && !drawer.querySelector('.story-drawer-drilldown')) {
        const host = drawer.querySelector('.dwrap') || drawer;
        const btn = doc.createElement('button');
        btn.type = 'button';
        btn.className = 'story-drawer-drilldown';
        btn.textContent = '深入调查该异常 · 对比 Runtime Trace 与静态 Pipeline →';
        btn.addEventListener('click', ev => triggerDrilldown(ev, '选中任务'));
        host.prepend(btn);
      }
    };

    inject();
    const observer = new MutationObserver(() => inject());
    observer.observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['hidden'] });
    const stopPoll = pollUntil(() => {
      inject();
      return Boolean(doc.querySelector('#lead-finding .story-drilldown-btn'));
    }, 6000, 200);

    return () => {
      observer.disconnect();
      stopPoll();
    };
  }

  /* 桥接 4：CSA 深入调查页，自动进入 Explorer，并在 Assignment 视角内直接展示修改意见与“修改代码”CTA */
  function bridgeCsaExplorer(win, doc, targetTab) {
    currentCsaTab = targetTab || 'pipeline';

    const styleId = 'story-csa-fix-card-style';
    if (!doc.getElementById(styleId)) {
      const style = doc.createElement('style');
      style.id = styleId;
      style.textContent = `
        .story-fix-card {
          position: fixed;
          right: 20px;
          bottom: 20px;
          width: min(370px, calc(100vw - 40px));
          z-index: 9999;
          padding: 14px 15px;
          border-radius: 12px;
          border: 1px solid rgba(255, 122, 89, 0.52);
          background: rgba(20, 23, 30, 0.94);
          backdrop-filter: blur(14px);
          box-shadow: 0 14px 36px rgba(0, 0, 0, 0.48);
          color: #eef1f6;
          font-family: Inter, -apple-system, "PingFang SC", sans-serif;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .story-fix-card[hidden] { display: none; }
        .story-fix-card__head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
        }
        .story-fix-card__kicker {
          font-size: 11px;
          font-family: "SFMono-Regular", Consolas, monospace;
          color: #ff9c82;
          letter-spacing: 0.02em;
        }
        .story-fix-card__close {
          border: 0;
          background: transparent;
          color: #9ba3b0;
          font-size: 16px;
          line-height: 1;
          padding: 2px 5px;
          cursor: pointer;
          border-radius: 4px;
        }
        .story-fix-card__close:hover {
          background: rgba(255, 255, 255, 0.08);
          color: #fff;
        }
        .story-fix-card h3 {
          margin: 0;
          font-size: 14px;
          font-weight: 600;
          color: #fff;
        }
        .story-fix-card p {
          margin: 0;
          font-size: 12.5px;
          line-height: 1.55;
          color: #c2cad6;
        }
        .story-fix-card__plan {
          padding: 8px 10px;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(255, 255, 255, 0.08);
          font-size: 12px;
          line-height: 1.5;
          color: #dbe1eb;
        }
        .story-fix-card__plan code {
          font-family: "SFMono-Regular", Consolas, monospace;
          font-size: 11.5px;
          padding: 1px 4px;
          border-radius: 4px;
          background: rgba(255, 255, 255, 0.09);
        }
        .story-fix-card__actions {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
          margin-top: 2px;
        }
        .story-fix-card__cta {
          flex: 1;
          padding: 8px 14px;
          border-radius: 8px;
          border: 1px solid #ff8769;
          background: #ff7a59;
          color: #140c0a;
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.14s ease, transform 0.14s ease;
        }
        .story-fix-card__cta:hover {
          background: #ff9175;
          transform: translateY(-1px);
        }
        .story-fix-card__pill {
          position: fixed;
          right: 20px;
          bottom: 20px;
          z-index: 9999;
          padding: 8px 13px;
          border-radius: 999px;
          border: 1px solid rgba(255, 122, 89, 0.65);
          background: rgba(22, 25, 32, 0.94);
          color: #ffd7cc;
          font: 500 12.5px/1.2 Inter, "PingFang SC", sans-serif;
          cursor: pointer;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }
        .story-fix-card__pill[hidden] { display: none; }
      `;
      doc.head.append(style);
    }

    let cardDismissed = false;
    let fixCard = doc.getElementById('story-assignment-fix-card');
    let reopenPill = doc.getElementById('story-assignment-fix-pill');

    if (!fixCard) {
      fixCard = doc.createElement('aside');
      fixCard.id = 'story-assignment-fix-card';
      fixCard.className = 'story-fix-card';
      fixCard.hidden = true;
      fixCard.innerHTML = `
        <div class="story-fix-card__head">
          <span class="story-fix-card__kicker">ASSIGNMENT 根因与修改意见</span>
          <button type="button" class="story-fix-card__close" aria-label="收起修改意见">×</button>
        </div>
        <h3>检测到 24 核组分核不均（12 慢 / 12 快）</h3>
        <p>Pipeline 层的 2.5× 耗时双峰并非流水线排布问题，而是 Assignment 层的 <code>qk_order</code> 映射碰撞，导致一半核组集中承载高代价压缩块。</p>
        <div class="story-fix-card__plan">
          <strong>修改意见：</strong>保持计算图拓扑不变，将 <code>qk_order</code> 改为奇偶错峰 / 轮转均衡映射，打散 24 核组高代价压缩块分布。
        </div>
        <div class="story-fix-card__actions">
          <button type="button" class="story-fix-card__cta">修改代码</button>
        </div>
      `;
      doc.body.append(fixCard);

      reopenPill = doc.createElement('button');
      reopenPill.id = 'story-assignment-fix-pill';
      reopenPill.type = 'button';
      reopenPill.className = 'story-fix-card__pill';
      reopenPill.hidden = true;
      reopenPill.textContent = '查看 Assignment 修改意见 · 修改代码';
      doc.body.append(reopenPill);

      fixCard.querySelector('.story-fix-card__close').addEventListener('click', () => {
        cardDismissed = true;
        fixCard.hidden = true;
        reopenPill.hidden = false;
      });

      reopenPill.addEventListener('click', () => {
        cardDismissed = false;
        reopenPill.hidden = true;
        fixCard.hidden = false;
      });

      fixCard.querySelector('.story-fix-card__cta').addEventListener('click', () => {
        triggerModifyCode({ source: 'html-cta', suggestionIds: ['assign-balance'] });
      });
    }

    const updateFixCardVisibility = tab => {
      const show = tab === 'assignment';
      fixCard.hidden = !show || cardDismissed;
      reopenPill.hidden = !show || !cardDismissed;
    };

    const syncFromCsa = () => {
      const st = win.PtoCsaExplorer?.getState?.();
      if (!st?.tab) return;
      updateFixCardVisibility(st.tab);
      if (st.tab === 'assignment' && activeViewId !== 'csa-assignment') {
        currentCsaTab = 'assignment';
        activeViewId = 'csa-assignment';
        renderStageChrome(VIEWS['csa-assignment']);
      } else if (st.tab === 'pipeline' && activeViewId !== 'csa-pipeline') {
        currentCsaTab = 'pipeline';
        activeViewId = 'csa-pipeline';
        renderStageChrome(VIEWS['csa-pipeline']);
      }
    };

    const onViewChange = () => syncFromCsa();
    doc.addEventListener('csa-view-change', onViewChange);

    const stopPoll = pollUntil(() => {
      const explorer = win.PtoCsaExplorer;
      if (!explorer) return false;
      if (!explorer.getState?.().open) {
        explorer.open();
      }
      const tabBtn = doc.querySelector(`[data-csa-tab="${currentCsaTab}"]`);
      if (!tabBtn) return false;
      if (tabBtn.getAttribute('aria-selected') !== 'true') {
        tabBtn.click();
      }
      syncFromCsa();
      return true;
    }, 7000, 150);

    return () => {
      doc.removeEventListener('csa-view-change', onViewChange);
      stopPoll();
    };
  }

  function switchCsaTab(tab) {
    const nextViewId = tab === 'assignment' ? 'csa-assignment' : 'csa-pipeline';
    currentCsaTab = tab;
    activeViewId = nextViewId;
    renderStageChrome(VIEWS[nextViewId]);
    try {
      const doc = stageFrame.contentDocument;
      const win = stageFrame.contentWindow;
      if (win?.PtoCsaExplorer && !win.PtoCsaExplorer.getState?.().open) {
        win.PtoCsaExplorer.open();
      }
      const btn = doc?.querySelector(`[data-csa-tab="${tab}"]`);
      if (btn) {
        btn.click();
      }
      const fixCard = doc?.getElementById('story-assignment-fix-card');
      const reopenPill = doc?.getElementById('story-assignment-fix-pill');
      if (fixCard) {
        const show = tab === 'assignment';
        fixCard.hidden = !show;
        if (reopenPill) reopenPill.hidden = true;
      }
      if (btn) return;
    } catch {}
    openView(nextViewId, { fromLane: cameFromLane });
  }

  function attachBridgeForView(view) {
    clearBridge();
    let win;
    let doc;
    try {
      win = stageFrame.contentWindow;
      doc = stageFrame.contentDocument;
    } catch {
      return;
    }
    if (!win || !doc) return;

    if (view.id === 'decision' || view.id === 'decision-v2') {
      bridgeCleanup = bridgeDecisionPage(win, doc);
    } else if (view.id === 'static' || view.id === 'static-v2') {
      bridgeCleanup = bridgeStaticExplorer(win, doc);
    } else if (view.id === 'lane') {
      bridgeCleanup = bridgePerfLane(win, doc);
    } else if (view.id === 'csa-pipeline' || view.id === 'csa-assignment') {
      bridgeCleanup = bridgeCsaExplorer(win, doc, view.csaTab);
    }
  }

  function openView(viewId, opts = {}) {
    const view = VIEWS[viewId];
    if (!view) return;

    const isCsaSameDoc =
      (viewId === 'csa-pipeline' || viewId === 'csa-assignment') &&
      loadedUrl === view.url &&
      stageFrame.contentWindow?.PtoCsaExplorer;

    activeViewId = viewId;
    cameFromLane = Boolean(opts.fromLane);
    if (view.csaTab) currentCsaTab = view.csaTab;

    highlightActiveCard(view.cardKey);

    setSidebar(false);
    if (shell.dataset.layout === 'chat-only') {
      setLayout('split');
    } else {
      setLayout(shell.dataset.layout);
    }

    renderStageChrome(view);

    if (isCsaSameDoc) {
      switchCsaTab(view.csaTab);
      return;
    }

    if (loadedUrl === view.url && stageFrame.contentDocument?.readyState === 'complete') {
      stageLoading.hidden = true;
      attachBridgeForView(view);
      return;
    }

    clearBridge();
    stageLoading.hidden = false;
    loadedUrl = view.url;

    const onLoad = () => {
      stageFrame.removeEventListener('load', onLoad);
      stageLoading.hidden = true;
      attachBridgeForView(view);
    };
    stageFrame.addEventListener('load', onLoad);
    stageFrame.src = view.url;
  }

  /* ========== Event Wiring ========== */
  document.addEventListener('change', ev => {
    if (ev.target.matches('input[name="perf-suggestion"]')) {
      syncSuggestionsUi();
    }
  });

  applyChatSuggestionsBtn.addEventListener('click', () => {
    const selectedIds = getSelectedSuggestionIds();
    if (!selectedIds.length) return;
    triggerModifyCode({ source: 'chat-suggestion', suggestionIds: selectedIds });
  });

  document.addEventListener('click', ev => {
    const openTrigger = ev.target.closest('[data-open-view]');
    if (openTrigger) {
      const targetId = openTrigger.dataset.openView;
      openView(targetId, { fromLane: false });
      return;
    }

    const scrollTrigger = ev.target.closest('[data-scroll-to]');
    if (scrollTrigger) {
      const target = document.getElementById(scrollTrigger.dataset.scrollTo);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }

    const navTrigger = ev.target.closest('[data-nav]');
    if (navTrigger) {
      document.querySelectorAll('[data-nav]').forEach(n => n.classList.remove('is-active'));
      navTrigger.classList.add('is-active');
      const key = navTrigger.dataset.nav;
      if (key !== 'threads') {
        showToast('当前聚焦于算子开发与性能诊断主线程');
      }
      return;
    }

    const threadTrigger = ev.target.closest('[data-thread]');
    if (threadTrigger) {
      const key = threadTrigger.dataset.thread;
      if (key !== 'current') {
        showToast('该历史线程已归档，当前保持在主诊断线程');
      } else {
        chatScroll.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }
  });

  backToLaneBtn.addEventListener('click', () => {
    openView('lane', { fromLane: false });
  });

  toggleChatSplitBtn.addEventListener('click', () => {
    setLayout(shell.dataset.layout === 'stage-only' ? 'split' : 'stage-only');
  });

  toggleCanvasBtn.addEventListener('click', () => {
    if (shell.dataset.layout === 'chat-only') {
      setSidebar(false);
      setLayout('split');
    } else {
      setLayout('chat-only');
    }
  });

  closeStageBtn.addEventListener('click', () => {
    setLayout('chat-only');
    setSidebar(true);
  });

  sidebarCollapseBtn.addEventListener('click', () => setSidebar(false));
  sidebarOpenBtn.addEventListener('click', () => setSidebar(true));

  newThreadBtn.addEventListener('click', () => {
    chatScroll.scrollTo({ top: 0, behavior: 'smooth' });
    showToast('已回到当前线程顶部');
  });

  composerForm.addEventListener('submit', ev => {
    ev.preventDefault();
    const text = composerInput.value.trim();
    if (!text) return;
    composerInput.value = '';

    const userMsg = document.createElement('article');
    userMsg.className = 'msg msg--user';
    userMsg.innerHTML = `<div class="msg__role">你</div><div class="msg__bubble"><p></p></div>`;
    userMsg.querySelector('p').textContent = text;

    const agentMsg = document.createElement('article');
    agentMsg.className = 'msg msg--agent';
    agentMsg.innerHTML = `
      <div class="msg__role">
        <span class="agent-badge">Codex</span>
        <span class="msg__time">已收到后续指令</span>
      </div>
      <div class="brief">
        <h2>下一轮验证准备</h2>
        <p>已记录你的跟进要求，完成 v2 上板运行后可直接拖入新的运行产物包，继续对比两版泳道与 Assignment 分布变化。</p>
      </div>
    `;

    chatThread.append(userMsg, agentMsg);
    chatScroll.scrollTo({ top: chatScroll.scrollHeight, behavior: 'smooth' });
  });

  syncSuggestionsUi();
})();
