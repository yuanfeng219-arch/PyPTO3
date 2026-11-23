/* 摘自 Design/PyPTO-Compute-Explorer-UI/js/ai-infra-ui.js 第 1–77 行（2026-11-10）：只保留 PtoAIInfraPresentation 图元皮肤，去掉依赖 IDE 外壳的页面级脚本。 */
/* UI-only adapter: no dispatch, camera writes, model edits or gesture handlers. */
(function () {
  'use strict';
  // Original demo vocabulary is authoritative.
  function localizeText(value) {
    // Preserve the original demo vocabulary; presentation never renames domain UI.
    return value;
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

  function operationGlyph(id,label,x,y,w,h,slot,options={}) {
    const shortLabel=label.replace('Context Accumulate','Accumulate');
    const key=options.iconKey||(/QK|PV/.test(label)?'matmul':/Softmax/.test(label)?'softmax':'reduce');
    const icon=window.PtoOpIcons.find(entry=>entry.id===key);
    const node=window.VisualSystem.node('op',id,shortLabel,0,0)
      .replace('width="260"','width="'+w+'"').replace('height="48"','height="34"')
      .replace('rx="24"','rx="17"').replace('x="130"','x="'+(w/2+8)+'"').replace('y="24"','y="17"').replace('font-size="18"','style="font-size:12px"');
    const badge=icon?`<g class="ai-operation-category" ><circle cx="16" cy="17" r="9" fill="#222"/><path transform="translate(10 11) scale(.5)" d="${icon.path}" fill="none" stroke="#eee" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></g>`:'';
    return `<g class="ai-operation-glyph" data-op-category="${key}" pointer-events="none" transform="translate(${x} ${y})">${node}${badge}<circle cx="8" cy="48" r="3" fill="var(--slot-color)"/><text class="ai-operation-slot" x="${w/2}" y="48" text-anchor="middle">${options.metadata||`Slot ${slot}`}</text></g>`;
  }

  window.PtoAIInfraPresentation = Object.freeze({ apply, iconFor, semanticRole, taskGlyph, operationGlyph, localizeText });

})();
