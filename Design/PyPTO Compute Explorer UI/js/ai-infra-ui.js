/* UI-only adapter: no dispatch, camera writes, model edits or gesture handlers. */
(function () {
  'use strict';
  // Presentation-only Chinese vocabulary. Source code and semantic IDs stay intact.
  const uiTerms = [
    ['QKV Preparation','QKV 准备'],['Aux Inputs','辅助输入'],['scaled / masked','缩放／掩码后的'],['PV partial','PV 部分结果'],['Key Cache','键缓存'],['Value Cache','值缓存'],['Cache Append','缓存追加'],['Mask','掩码'],['Projection','投影'],['Scaling','缩放'],['Preparation','准备'],['Inputs','输入'],['Outputs','输出'],['Static','静态'],['Producer','生产者'],['Consumer','消费者'],['Buffer','缓冲区'],['Feedback','反馈'],['rescale','重缩放'],
    ['Attention+Output','注意力与输出'],['Attention Tasks + Output','注意力任务与输出'],
    ['Work Generation','工作项生成'],['Task Generation','任务生成'],['Block Assignment','执行块分配'],['Active Blocks','活跃执行块'],
    ['Static Execution Weight','静态执行次数'],['Execution Weight','执行次数'],['Fit to View','适应视图'],
    ['Logical Tick','逻辑步'],['Task Inputs','任务输入'],['Task Output','任务输出'],['Tile Shape','分配形状'],['Valid Shape','有效形状'],
    ['Q Heads','Q 头'],['KV Head','KV 头'],['Q Head','Q 头'],['Q Tile','Q 数据块'],['K Cache','K 缓存'],['V Cache','V 缓存'],['KV Cache','KV 缓存'],
    ['Context Accumulate','上下文累加'],['Online Softmax','在线归一化'],['Output Normalize','输出归一化'],['Attention Output','注意力输出'],
    ['Scaling + Mask','缩放与掩码'],['QK Matmul','QK 矩阵乘'],['PV Matmul','PV 矩阵乘'],['Partial Context','部分上下文'],
    ['Update Maximum','更新最大值'],['Rescale FIFO','重缩放队列'],['Probability','概率'],['Scores','得分'],
    ['Assignment','任务分配'],['Pipeline','流水线'],['Graph','计算图'],['Structure','结构'],['Phase0','准备阶段'],
    ['Normalize','归一化'],['Accumulate','累加'],['Store','写回'],['Append','追加'],['Feedback','反馈'],['State','状态'],['Data','数据'],
    ['Batches','批次'],['Batch','批次'],['Blocks','执行块'],['Block','执行块'],['Tasks','任务'],['Task','任务'],
    ['Stacks','分块'],['Stack','分块'],['Pages','页'],['Page','页'],['Slots','槽位'],['Slot','槽位'],
    ['BATCH','批次'],['BLOCK','执行块'],['token','词元'],['Head','头'],['Norm','归一化'],['MatMul','矩阵乘'],['Matmul','矩阵乘'],['Reduce','归约'],['Maximum','最大值'],['Update','更新'],['Context','上下文'],['Output','输出'],['Input','输入'],['Cache','缓存'],['Shape','形状'],['Precision','精度'],['Memory','存储'],['Valid','有效'],['Allocated','分配'],['Cast','类型转换'],
    ['tokens','词元'],['Items','工作项'],['Per','每'],['Lane','通道'],['Tick','逻辑步']
  ];
  const objectTerms = new Set(['Batch','Batches','Block','Blocks','Task','Tasks','Stack','Stacks','Page','Pages','Slot','Slots','Q Heads','KV Head','Q Head','Q Tile','Head','BATCH','BLOCK']);
  function localizeText(value) {
    let result=value;
    for(const [english,chinese] of uiTerms.filter(([term])=>!objectTerms.has(term))) result=result.replace(new RegExp('\\b'+english.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'\\b','g'),chinese);
    return result;
  }
  function localizeUI(scope) {
    if(!(scope instanceof Element)) return;
    const walker=document.createTreeWalker(scope,NodeFilter.SHOW_TEXT);
    let node;
    while((node=walker.nextNode())){
      if(node.parentElement?.closest('script,style,pre,code,.ai-presentation-overlay,.viz2-header .pto-ide-frame__pane-title,.viz2-graph-scene,.viz2-graph-stage-title,.viz2-graph-block-rail,.ai-semantic-legend,#dslEditor,#sourceRegionBar,[data-compute-code],[data-intra-code]')) continue;
      const translated=localizeText(node.nodeValue); if(translated!==node.nodeValue) node.nodeValue=translated;
    }
    for(const element of [scope,...scope.querySelectorAll('[aria-label],[data-ai-tip]')].filter(el=>!el.closest('.viz2-graph-scene,.viz2-graph-block-rail'))) for(const attr of ['aria-label','data-ai-tip']){
      const old=element.getAttribute(attr); if(old){const next=localizeText(old);if(next!==old)element.setAttribute(attr,next);}
    }
  }

  function apply(scene, graph, options) {
    const layers = Object.fromEntries(['shape','precision','memory','control'].map(key =>
      [key, scene.dataset['overlay' + key[0].toUpperCase() + key.slice(1)] === 'true']));
    // The approved Pattern renderer changes the node contents only. Keep the
    // original delegated expansion target and routed dimensions intact.
    window.PtoStudioGraphLegendSkin.apply(scene, graph, { layers });
    for (const node of graph.nodes) {
      const element = [...scene.querySelectorAll('[data-pypa-node]')].find(el => el.dataset.pypaNode === node.id);
      if (!element) continue;
      const toggle = element.querySelector('.legend-toggle');
      if (toggle && node.expandable) {
        toggle.dataset.pypaExpandToggle = node.id;
        toggle.textContent = options.expandedStages?.has(node.id) ? '−' : '+';
        toggle.setAttribute('aria-hidden', 'true');
      }
    }
  }
  function semanticRole(node) {
    const data=node.data || node.semantic?.data || node.sourceNode?.data || {};
    const id=String(node.id), label=String(node.title||'');
    const palette=window.VisualSystem.tokens.role;
    if (data.role==='weight') return {key:'weight',color:palette.weight};
    if (data.role==='gradient') return {key:'gradient',color:palette.gradient};
    if (data.role==='optimizer') return {key:'optimizer',color:palette.optimizer};
    if (/sum|max|accum|rescale|context|scores|partial/i.test(id+' '+label)) return {key:'aggregate',color:palette.aggregate};
    if (/aux|metadata|slot_mapping|block_table|seq_lens/i.test(id+' '+label)) return {key:'metadata',color:palette.optimizer};
    return {key:'activation',color:'#80B7F7'};
  }
  function iconFor(node) {
    const label = node.title || '';
    const rules = [['exchange', /all.?gather|all.?to.?all|reduce.?scatter|notify|publish/i], ['norm', /norm/i], ['rope', /rope/i], ['topk', /top.?k|indexer/i], ['compress', /compress/i], ['reshape', /reshape|view|cast/i], ['concat', /concat/i], ['split', /split/i], ['mix', /hc.?pre|hc.?post/i], ['linear', /projection/i]];
    const id = rules.find(([,pattern]) => pattern.test(label))?.[0];
    return window.PtoOpIcons.find(icon => icon.id === id) || null;
  }
  function taskGlyph(id,w,h) {
    // Canonical Task identity, compact representation; capacity stays in the
    // existing segmented encoding. Selection is separate from object identity.
    const source = window.VisualSystem.node('task', 'task-'+id, '', 0, 0);
    const side = Math.min(14, h - 6);
    return `<svg class="ai-task-identity" pointer-events="none" x="4" y="${(h-side)/2}" width="${side}" height="${side}" viewBox="10 12 36 36" preserveAspectRatio="xMidYMid meet" overflow="hidden">${source}</svg>`;
  }

  function operationGlyph(id,label,x,y,w,h,slot) {
    const shortLabel=label.replace('Context Accumulate','Accumulate');
    const key=/QK|PV/.test(label)?'matmul':/Softmax/.test(label)?'softmax':'reduce';
    const icon=window.PtoOpIcons.find(entry=>entry.id===key);
    const node=window.VisualSystem.node('op',id,shortLabel,0,0)
      .replace('width="260"','width="'+w+'"').replace('height="48"','height="34"')
      .replace('rx="24"','rx="17"').replace('x="130"','x="'+(w/2+8)+'"').replace('y="24"','y="17"').replace('font-size="18"','style="font-size:12px"');
    const badge=icon?`<g class="ai-operation-category" transform="translate(6 7) scale(.8)"><circle cx="12" cy="12" r="12" fill="#222"/><path d="${icon.path}" fill="none" stroke="#eee" stroke-width="1.5"/></g>`:'';
    return `<g class="ai-operation-glyph" pointer-events="none" transform="translate(${x} ${y})">${node}${badge}<circle cx="8" cy="48" r="3" fill="var(--slot-color)"/><text class="ai-operation-slot" x="${w/2}" y="48" text-anchor="middle">Slot ${slot}</text></g>`;
  }

  window.PtoAIInfraPresentation = Object.freeze({ apply, iconFor, semanticRole, taskGlyph, operationGlyph, localizeText });

  function transport(button, replay=false) {
    const label = replay ? '重播' : button.textContent.trim();
    if (!replay && !['播放','暂停'].includes(label)) return;
    button.classList.add('ai-icon-button');
    button.setAttribute('aria-label',label); button.dataset.aiTip=label;
    button.dataset.playing=String(label==='暂停');
    if (replay) { if (!button.querySelector('svg')) button.innerHTML=window.AIInfraTransportIcons.restart; }
    else { button.setAttribute('aria-pressed',String(label==='暂停')); button.innerHTML=window.AIInfraTransportIcons.playback; }
  }
  function organizeWorkspace() {
    const root=document.querySelector('.viz2:not([hidden])'); if (!root) return;
    const tabs=root.querySelector('.viz2-tabs .tab-control'); if(tabs) tabs.dataset.liquidGlass='control';
    root.querySelectorAll('[data-viz2-scope],[data-pipeline-scope],[data-graph-scope]').forEach(button=>button.parentElement.remove());
    root.querySelectorAll('.viz2-anatomy-card > header h3').forEach(title=>{
      if (title.dataset.aiTaskGlyph) return;
      const label=localizeText(title.textContent.trim());
      let glyph=window.VisualSystem.node('task',label,label,0,0).replace('fill="#9EF04B"','fill="#292929"').replace('fill="#080b05" text-anchor','fill="#e8e8e8" text-anchor');
      title.style.backgroundImage='url("data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 60">'+glyph+'</svg>')+'")';
      title.dataset.aiTaskGlyph='true';
    });
    root.querySelectorAll('#viz2-graph-inter,#viz2-graph-intra').forEach(host=>{
      if(host.querySelector('.ai-semantic-legend')) return;
      const legend=document.createElement('div');legend.className='ai-semantic-legend';legend.setAttribute('aria-label','图元语义色');
      legend.innerHTML='<span><i style="--role-color:#80B7F7"></i>Activation / KV Data</span><span><i style="--role-color:#F2C45F"></i>Intermediate / Reduction</span><span><i style="--role-color:#B0B0B0"></i>Metadata</span>';host.append(legend);
    });
    root.querySelectorAll('[data-am-play],[data-p0-play]').forEach(b=>transport(b));
    root.querySelectorAll('[data-am-replay],[data-p0-replay]').forEach(b=>transport(b,true));
    // Reparent only within the owning controller: delegated click handlers,
    // direct listeners and live readouts remain attached to the original nodes.
    root.querySelectorAll('.viz2-am-head').forEach(head=>{
      const owner=head.parentElement, footer=owner.querySelector('.viz2-am-footer'), controls=head.querySelector('.viz2-am-controls');
      if (footer && controls) { controls.setAttribute('role','group'); controls.setAttribute('aria-label','播放与分配'); footer.prepend(controls); }
    });
    for (const panel of root.querySelectorAll('.viz2-canvas:not([hidden])')) {
      const footers=[...panel.querySelectorAll('footer')].filter(f=>f.getClientRects().length);
      const footer=footers[0]; if (!footer) continue;
      footer.classList.add('ai-control-bar'); footer.removeAttribute('data-liquid-glass'); footer.setAttribute('aria-label','画布控制栏');
      let dock=footer.querySelector('.ai-operation-dock');
      if(!dock){dock=document.createElement('div');dock.className='ai-operation-dock';dock.setAttribute('role','group');dock.setAttribute('aria-label','操作组件');footer.prepend(dock);}
      dock.dataset.liquidGlass='control';
      const computeToolbar=[...panel.querySelectorAll('[data-compute-toolbar]')].find(n=>n.getClientRects().length);
      if (computeToolbar && computeToolbar.parentElement!==dock) dock.prepend(computeToolbar);
      const back=[...panel.querySelectorAll('[data-intra-back],[data-anatomy-back]')].find(n=>n.getClientRects().length);
      if (back) { back.classList.add('ai-detail-back'); back.dataset.liquidGlass='control'; if (back.parentElement!==panel) panel.prepend(back); }
      const tasknav=[...panel.querySelectorAll('.viz2-anatomy-task-nav')].find(n=>n.getClientRects().length);
      if (tasknav && tasknav.parentElement!==dock) dock.prepend(tasknav);
      const phase=panel.querySelector('.viz2-phase-toolbar');
      if (phase && phase.parentElement!==dock) dock.prepend(phase);
      footer.querySelectorAll('.kf-pa-map__tools').forEach(tools=>{if(tools.parentElement!==footer)footer.append(tools);tools.dataset.liquidGlass='control';tools.setAttribute('role','group');tools.setAttribute('aria-label','缩放组件');});
      footer.querySelectorAll('.viz2-am-controls,.viz2-graph-footer__controls').forEach(group=>{if(group.parentElement!==dock)dock.append(group);});
      dock.querySelectorAll('[data-liquid-glass]').forEach(child=>child.removeAttribute('data-liquid-glass'));
      const hideDock=!dock.querySelector('button,select,summary'); if(dock.hidden!==hideDock) dock.hidden=hideDock;
      footer.querySelectorAll('.segmented-control').forEach(group=>{ if (!group.getAttribute('aria-label')) group.setAttribute('aria-label','分配状态'); });
    }
  }
  let organizing=false;
  function scheduleWorkspace() { if (organizing) return; organizing=true; requestAnimationFrame(()=>{ organizeWorkspace(); organizing=false; }); }
  new MutationObserver(scheduleWorkspace).observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['hidden']});
  scheduleWorkspace();

  const glassGroups = '.viz2-tabs .tab-control,.segmented-control,.kf-pa-map__tools,.csa-qk-zoom,.csa-qp-footer-tools,.viz2-graph-footer__controls';
  const glassPanels = '.viz2-graph-inspector,.viz2-p0-inspector,.viz2-anatomy-inspector,.csa-qp-inspector,.csa-inspector,.csa-qk-drawer,.viz2-graph-layer-menu__content,.viz2-graph-block-rail';
  function prepare(scope) {
    if (!(scope instanceof Element)) return;
    if (!scope.matches('.viz2,.csa-explorer') && !scope.closest('.viz2,.csa-explorer')) {
      scope.querySelectorAll('.viz2,.csa-explorer').forEach(prepare); return;
    }
    for (const [selector,preset] of [[glassGroups,'control'],[glassPanels,'panel']]) {
      const elements = [...scope.querySelectorAll(selector)];
      if (scope.matches(selector)) elements.push(scope);
      for (const element of elements) {
        // Nested refraction would sample the same pixels twice.
        if (!element.parentElement?.closest('[data-liquid-glass]') && element.dataset.liquidGlass !== preset)
          element.dataset.liquidGlass = preset;
      }
    }
    const titled = [...scope.querySelectorAll('[title]')];
    if (scope.hasAttribute('title')) titled.push(scope);
    for (const element of titled) {
      element.dataset.aiTip = element.getAttribute('title');
      element.removeAttribute('title');
    }
    const named = [...scope.querySelectorAll('[data-viz2-graph-layer],[data-qp-zoom],[data-qp-map-zoom],[data-qp-fit],[data-qp-map-fit]')];
    for (const element of named) {
      const zoom = element.dataset.qpZoom ?? element.dataset.qpMapZoom;
      element.setAttribute('aria-label', zoom != null ? (Number(zoom) < 1 ? '缩小' : '放大') : element.textContent.trim());
    }
  }
  prepare(document.body);
  const observer = new MutationObserver(records => {
    for (const record of records) {
      if (record.type === 'attributes') prepare(record.target);
      else for (const node of record.addedNodes) prepare(node);
    }
  });
  observer.observe(document.body, { childList:true, subtree:true, attributes:true, attributeFilter:['title'] });

  const help = document.createElement('button');
  help.type = 'button'; help.className = 'ai-help';
  help.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v1"/></svg>';
  help.dataset.liquidGlass='icon'; help.setAttribute('aria-label','使用说明'); help.setAttribute('aria-expanded','false'); help.setAttribute('aria-controls','ai-usage-panel');
  const attachHelp = () => { const header = document.querySelector('.viz2:not([hidden]) .viz2-header,.csa-explorer:not([hidden]) header'); if (header && !header.contains(help)) header.append(help); };
  new MutationObserver(attachHelp).observe(document.querySelector('#ideWorkarea'), {childList:true,subtree:true,attributes:true,attributeFilter:['hidden']});
  attachHelp();
  const dialog = document.createElement('aside'); dialog.className = 'ai-usage'; dialog.id='ai-usage-panel'; dialog.hidden=true; dialog.setAttribute('role','region');
  dialog.setAttribute('aria-labelledby','ai-usage-title');
  dialog.innerHTML = `<header><h2 id="ai-usage-title">Compute Explorer · 使用说明</h2><button type="button" data-ai-close aria-label="关闭使用说明">×</button></header>
    <h3>价值</h3><p>沿同一份算子源码，查看数据与任务归属、流水结构和计算依赖。本页直接查看 Paged Attention；原 IDE 入口保留源码与 Decode CSA 场景。</p>
    <h3>Graph 同步事件</h3><p>QK_READY、SOFTMAX_READY 与 PV_READY 标记计算和搬运之间的 SET／WAIT 依赖。它们表示源码中的同步顺序，不表示运行耗时；可在图层菜单中控制同步关系的显示。</p>
    <h3>流水线图例</h3><p>准备阶段：每排对应一条 AIV Lane，每个正方形对应一个 Batch × KV Head 预处理工作项。方格表示工作归属，不表示数据量或执行耗时。未选中时去色，选中对应 Block 时恢复语义色。</p>
    <h3>任务分配图例</h3><p>任务矩阵：一格 = Q Head，每个 Task 固定 5 个。Block 分组：一格 = Stack，每个 Stack 最多 512 tokens。Batch 侧方格表示 Stack。双击 Task 查看内部；此视图显示静态映射，不表示执行时序。</p>
    <h3>使用</h3><p>选择 Assignment、Pipeline 或 Graph。点击对象查看详情，展开模块或进入 Block 内部；通过源码入口回到具体文件与行段。Assignment 支持拖动平移与滚轮缩放；Pipeline 总览纵向滚动浏览 Block，详情卡内滚动浏览 Stack；底栏切换 Task，适应按钮返回全览，返回按钮回到上一层。</p>
    <div class="ai-prompt"><button type="button" data-ai-copy aria-label="复制继续调查 prompt"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M15 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h3"/></svg></button><pre>使用 ai-infra-investigation，核查 PyPTO Compute Explorer 中的 [文件 / 视图 / 对象]。结合页面中的源码行段与参数验证其映射关系，保留现有交互，不修改算子。</pre></div><p data-ai-copy-status role="status"></p>
    <h3>扩展</h3><p>新增场景需接入源码快照、独立领域模型与对象映射；不能仅替换图上的名称。当前页面呈现源码与静态逻辑关系，运行耗时需要额外的设备数据。</p>
    <h3>实现分工</h3><p>既有模型与控制器负责对象、选择、布局、动画与视角；AI Infra Paged Attention Pattern 提供图元皮肤，Spatial Systems UI 提供设计变量和 Liquid Glass 控件，ai-infra-investigation 组织复用与验收。</p>`;
  document.body.append(dialog);
  const setHelpOpen = open => { dialog.hidden=!open; help.setAttribute('aria-expanded',String(open)); if (!open) help.focus(); };
  help.addEventListener('click', () => setHelpOpen(dialog.hidden));
  dialog.querySelector('[data-ai-close]').addEventListener('click', () => setHelpOpen(false));
  document.addEventListener('keydown', event => { if (event.key==='Escape' && !dialog.hidden) { event.preventDefault(); setHelpOpen(false); } });
  dialog.querySelector('[data-ai-copy]').addEventListener('click', async () => {
    const status = dialog.querySelector('[data-ai-copy-status]');
    try {
      await navigator.clipboard.writeText(dialog.querySelector('pre').textContent);
      status.textContent = '已复制';
    } catch { status.textContent = '复制失败，请选中正文手动复制。'; }
  });

  // Chrome uses one shared tooltip, including keyboard focus. Objects that
  // already own a domain tooltip keep that richer preview on pointer hover.
  const tooltip = document.createElement('div'); tooltip.className = 'ai-tooltip';
  tooltip.id = 'ai-ui-tooltip'; tooltip.role = 'tooltip'; tooltip.hidden = true;
  document.body.append(tooltip);
  let described = null;
  function hide() {
    tooltip.hidden = true;
    if (described) {
      const ids = (described.getAttribute('aria-describedby') || '').split(' ').filter(id => id && id !== tooltip.id);
      if (ids.length) described.setAttribute('aria-describedby', ids.join(' '));
      else described.removeAttribute('aria-describedby');
    }
    described = null;
  }
  document.addEventListener('toggle',event=>{if(event.target.matches?.('.viz2-graph-layer-menu')&&event.target.open)hide();},true);
  function show(event) {
    const selector = 'button,summary,select,input,[role="button"],[role="checkbox"]';
    const target = event.target.closest(event.type === 'focusin' ? selector + ',[tabindex][aria-label]' : selector);
    if (!target || target === described) return;
    if(target.matches('.viz2-graph-layer-menu[open] > summary')){hide();return;}
    if (!target.closest('.viz2,.csa-explorer,.ai-usage')) return;
    hide();
    const domainObject = target.closest('.kf-pa-map__viewport,.viz2-graph-viewport,.viz2-anatomy-svg,.viz2-intra-svg,.viz2-pipeline-viewport,.csa-qk-canvas-viewport');
    if (domainObject) return;
    const label = target.textContent.trim() || target.getAttribute('aria-label') || ''; 
    const layerHelp = {
      data:'显示或隐藏算子之间的张量节点，例如 Q/K/V、得分和概率；隐藏后保留算子及其依赖关系',
      shape:'在张量节点上显示源码声明的维度与有效范围',
      precision:'显示张量和算子输出的数据类型，例如 BF16、FP32',
      memory:'显示数据存储位置及算子所属的 AIC／AIV 执行单元',
      control:'显示状态传递、循环控制等非张量依赖',
      execution:'用堆叠和次数标记显示静态执行次数，不代表耗时',
      sync:'显示计算与搬运之间的同步关系'
    };
    const purpose = target.matches('[data-viz2-graph-layer]') ? layerHelp[target.dataset.viz2GraphLayer]
      : target.matches('.viz2-graph-layer-menu > summary') ? '选择要显示的形状、精度、存储、控制、执行次数和同步信息'
      : target.matches('[data-am-state="generation"]') ? '按 Batch 行与 KV Head 列查看 Task；每个 Task 包含 5 个 Q Head'
      : target.matches('[data-am-state="assignment"]') ? '将同一批 Task 按 Task ID % 24 重排，查看它们归属的执行 Block'
      : target.matches('[data-am-replay]') ? '从任务矩阵开始，重新播放一次分配过程'
      : target.matches('[data-am-play]') ? (label === '暂停' ? '暂停当前分配动画，保留当前位置' : '播放一次任务分配过程')
      : target.matches('[data-am-fit]') ? '恢复默认缩放与位置，显示完整任务布局'
      : target.matches('[data-am-zoom]') ? (Number(target.dataset.amZoom) < 1 ? '缩小布局，查看更多 Task' : '放大布局，查看方格细节')
      : target.matches('[data-am-state]') ? '' : null;
    const helpByLabel = {'Assignment':'查看 Task 生成及其到执行 Block 的映射','Pipeline':'查看各 Block 内任务的流水线与依赖','Graph':'查看计算节点、数据流及源码关联','Phase0':'查看准备阶段的任务分配','Attention+Output':'查看注意力与输出阶段的任务分配','图元':'展开所选 Task 的 Q Head、Stack 和 Page 结构说明'};
    const text = purpose || helpByLabel[label] || Object.entries(helpByLabel).find(([key])=>localizeText(key)===label)?.[1] || target.dataset.aiTip || target.getAttribute('aria-label') || label;
    if (!text) return;
    described = target;
    const previous = target.getAttribute('aria-describedby');
    if (!text || localizeText(text).trim() === localizeText(label).trim()) return;
    target.setAttribute('aria-describedby', [previous,tooltip.id].filter(Boolean).join(' '));
    tooltip.textContent = text; tooltip.hidden = false;
    // A modal dialog is in the top layer; its tooltip must share that layer.
    (target.closest('dialog') || document.body).append(tooltip);
    const box = target.getBoundingClientRect(), tip = tooltip.getBoundingClientRect();
    tooltip.style.left = Math.max(8,Math.min(innerWidth-tip.width-8,box.left+(box.width-tip.width)/2))+'px';
    tooltip.style.top = (box.bottom+tip.height+16<innerHeight?box.bottom+8:Math.max(8,box.top-tip.height-8))+'px';
  }
  document.addEventListener('pointerover',show);
  document.addEventListener('focusin',show);
  document.addEventListener('pointerout',event => { if (!event.relatedTarget || !described?.contains(event.relatedTarget)) hide(); });
  document.addEventListener('focusout',hide);
  document.addEventListener('pointerdown',hide,true);
  document.addEventListener('keydown',event => { if (event.key === 'Escape') hide(); });
// Localize only changed subtrees, once per frame; never rescan hidden views on hover.
(function(){
 let frame=0;const dirty=new Set();
 const add=node=>{const element=node instanceof Element?node:node.parentElement;if(element?.closest('.viz2,.ai-usage,.ai-tooltip'))dirty.add(element);};
 const refresh=()=>{frame=0;const scopes=[...dirty];dirty.clear();for(const scope of scopes)if(scope.isConnected&&!scopes.some(other=>other!==scope&&other.contains(scope)))localizeUI(scope);};
 new MutationObserver(records=>{for(const record of records){if(record.type==='characterData')add(record.target);else for(const node of record.addedNodes)add(node);}if(dirty.size&&!frame)frame=requestAnimationFrame(refresh);}).observe(document.body,{subtree:true,childList:true,characterData:true});
 document.querySelectorAll('.viz2,.ai-usage,.ai-tooltip').forEach(add);if(dirty.size)frame=requestAnimationFrame(refresh);
})();

})();

if (document.body.classList.contains('ai-extracted')) { window.addEventListener('load', () => document.querySelector('[data-visualization2-open]')?.click()); }

