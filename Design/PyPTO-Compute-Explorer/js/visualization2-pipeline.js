/* Static Block execution model; source: paged_attention_pypto.py 253–266, 864–872.
 * Task bodies encode Stack counts. Output caps are equal-size symbols, not durations.
 * No host dispatches, inter-Block Task edges, or post-attention gather are inferred.
 */
(function (global) {
  'use strict';

  function createModel({ blocks, kvHeads, seqLens, pageTokens, pagesPerStack, phase0Model }) {
    const stackTokens = pageTokens * pagesPerStack;
    const tasks = Array.from({ length: seqLens.length * kvHeads }, (_, id) => {
      const batch = Math.floor(id / kvHeads), kvHead = id % kvHeads, seqLen = seqLens[batch];
      return { id, block: id % blocks, batch, kvHead, seqLen,
        segments: global.PtoVisualization2TaskEncoding.segments(seqLen, stackTokens, pageTokens),
        pages: Math.ceil(seqLen / pageTokens), stacks: Math.ceil(seqLen / stackTokens),
        output: { taskId: id, normalize: true, dtype: 'BF16', store: true } };
    });
    const lanes = Array.from({ length: blocks }, (_, block) => {
      const assigned = tasks.filter(task => task.block === block);
      const phase0Lanes = phase0Model.lanes.filter(lane => lane.block === block);
      return { block, tasks: assigned, phase0Lanes,
        phase0Groups: phase0Lanes.reduce((sum, lane) => sum + lane.groups.length, 0),
        totalStacks: assigned.reduce((sum, task) => sum + task.stacks, 0) };
    });
    return { blocks, kvHeads, stackTokens, pageTokens, pagesPerStack, tasks, lanes };
  }

  function mount(host, { model, onSelect, onCamera, onEnter }) {
    const pattern = global.PtoSwimlaneTaskPattern;
    const colormap = pattern.createTaskColormap();
    // Token capacity uses the same blue as Assignment; operation caps retain the shared palette.
    const colors = { attention: getComputedStyle(host).getPropertyValue('--primary').trim(),
      output: colormap.colorForTask({ label: 'vec_reduce' }), phase0: colormap.colorForTask({ label: 'vec_layernorm' }) };
    host.classList.add('viz2-pipeline');
    host.innerHTML = `<div class="viz2-pipeline-body">
      <div class="viz2-pipeline-stage">
        <div class="viz2-pipeline-viewport" tabindex="0" role="group" aria-label="Block Execution Map，横向滚动缩放泳道，纵向滚动浏览 Block 行，拖拽横向平移">
          <div class="viz2-pipeline-surface"><canvas aria-hidden="true"></canvas><div class="viz2-pipeline-targets"></div></div>
        </div>
        <footer class="viz2-pipeline-footer"><span><i style="background:${colors.attention}"></i>Attention · 宽度 ∝ Stack 数 <i style="background:${colors.output}"></i>每 Task 末尾 Output · 等宽符号 <small>横向滚动缩放 · 纵向滚动浏览 Block</small></span><div class="kf-pa-map__tools"><button class="btn btn-ghost btn-sm" type="button" data-bem-zoom="0.8" aria-label="缩小 Pipeline">−</button><output data-bem-scale>100%</output><button class="btn btn-ghost btn-sm" type="button" data-bem-zoom="1.25" aria-label="放大 Pipeline">＋</button><button class="btn btn-ghost btn-sm" type="button" data-bem-fit>适应</button></div></footer>
      </div>
      <aside class="pto-ide-frame__pane viz2-pipeline-inspector" aria-label="Pipeline 执行详情" hidden>
        <header class="pto-ide-frame__pane-header"><b data-bem-title></b><button class="btn btn-ghost btn-sm" type="button" data-bem-clear aria-label="清除 Pipeline 选择">×</button></header>
        <div class="pto-ide-frame__pane-body" data-bem-detail></div>
      </aside>
    </div>`;
    const viewport = host.querySelector('.viz2-pipeline-viewport'), surface = host.querySelector('.viz2-pipeline-surface');
    const canvas = host.querySelector('canvas'), ctx = canvas.getContext('2d'), targets = host.querySelector('.viz2-pipeline-targets');
    const inspector = host.querySelector('aside'), detail = host.querySelector('[data-bem-detail]');
    const taskDescription = task => `Task ${task.id} · Block ${String(task.block).padStart(2, '0')} · Batch ${task.batch} · KV ${task.kvHead} · ${task.seqLen} tokens · ${task.stacks} Stacks`;
    targets.innerHTML = model.lanes.map(lane => `<button class="viz2-pipeline-target" type="button" data-bem-block="${lane.block}" aria-label="Pipeline Block ${String(lane.block).padStart(2,'0')}，${lane.tasks.length} Tasks" aria-pressed="false"></button>
      ${lane.tasks.map(task => `<button class="viz2-pipeline-target" type="button" data-bem-task="${task.id}" aria-label="${taskDescription(task)}，末尾 Output Normalize 和 BF16 Store" aria-pressed="false"></button>`).join('')}`).join('');
    let selection = null, camera = null, layout = null, drag = null, suppressClick = false, hover = null, active = false;
    let selectionKey = '', focusOnNextPaint = false, transitionMasked = false, transitionFocusBlock = null, semanticTaskMask = null, doubleCandidate = null;
    const tooltip = document.createElement('div');
    tooltip.className = 'pto-swimlane-task-tooltip viz2-pipeline-tooltip'; tooltip.setAttribute('role','tooltip');
    const tooltipController = pattern.initHoverTooltip({ root: viewport, targets: '[data-bem-task], [data-bem-block]',
      tooltip, appendTo: viewport, bounds: viewport,
      getTask: target => target.hasAttribute('data-bem-task') ? model.tasks[Number(target.dataset.bemTask)] : model.lanes[Number(target.dataset.bemBlock)],
      getTooltipHtml: item => item.id != null
        ? `<b>Task ${item.id} · Block ${item.block}</b><span>Batch ${item.batch} · KV Head ${item.kvHead}</span><span>${item.seqLen} tokens · ${item.stacks} Stacks</span><span>末尾：Normalize → BF16 Store</span>`
        : `<b>Block ${String(item.block).padStart(2,'0')}</b><span>Phase0 ${item.phase0Groups ? `${item.phase0Groups} 组预处理` : '无有效预处理'} · 参与 SYNCALL</span><span>Task ${item.tasks.map(task=>task.id).join(' → ')}</span>` });

    function measure() {
      host.style.setProperty('--viz2-pipeline-footer-height', `${host.querySelector('.viz2-pipeline-footer').offsetHeight}px`);
      const width = viewport.clientWidth, header = 40;
      const row = Math.max(36,(viewport.clientHeight-header-8)/model.blocks);
      const height = Math.max(viewport.clientHeight,header+row*model.blocks+8);
      const labelX = 12, phaseX = 72, phaseW = 98, syncX = 198, taskX = 232, outputW = 28, gap = 8;
      const available = Math.max(64, width - taskX - 18);
      const unit = Math.max(2, Math.min(...model.lanes.filter(lane=>lane.tasks.length).map(lane =>
        (available - lane.tasks.length * outputW - Math.max(0,lane.tasks.length-1) * gap) / Math.max(1,lane.totalStacks))));
      return { width, height, header, row, labelX, phaseX, phaseW, syncX, taskX, outputW, gap, unit };
    }
    function maxOverflow(scale) {
      if (!layout) return 0;
      return Math.max(0, ...model.lanes.map(lane => lane.totalStacks * layout.unit * scale + lane.tasks.length * layout.outputW + Math.max(0,lane.tasks.length-1)*layout.gap - (layout.width-layout.taskX-18)));
    }
    function bounded(next) { return { scale: next.scale, x: Math.max(-maxOverflow(next.scale), Math.min(0,next.x)) }; }
    function positions() {
      const c = camera || {scale:1,x:0};
      return model.lanes.flatMap(lane => {
        let x = layout.taskX + c.x;
        return lane.tasks.map(task => {
          const width = task.stacks * layout.unit * c.scale, y = layout.header + task.block * layout.row + 3;
          const position = { task, x, y, width, outputX:x+width, outputWidth:layout.outputW, height:layout.row-6 };
          x += width + layout.outputW + layout.gap; return position;
        });
      });
    }
    function revealTask() {
      if (selection?.kind !== 'task' || !layout || !camera) return;
      const rect = positions().find(item => item.task.id === selection.task);
      if (!rect) return;
      const left = layout.taskX + 2, right = layout.width - 12;
      if (rect.x >= left && rect.outputX + rect.outputWidth <= right) return;
      const next = bounded({...camera,x:camera.x + (left+right)/2 - (rect.x+rect.outputX+rect.outputWidth)/2});
      if (Math.abs(next.x-camera.x)>.5) { camera=next; onCamera(next); }
    }
    const color = name => getComputedStyle(host).getPropertyValue(name).trim();
    function place(element, x, y, width, height) {
      Object.assign(element.style, {left:`${x}px`,top:`${y}px`,width:`${width}px`,height:`${height}px`});
    }
    function paint() {
      if (!active || !viewport.clientWidth || !viewport.clientHeight) return;
      layout = measure();
      if (!camera) { camera={scale:1,x:0}; onCamera(camera); return; }
      const next = bounded(camera);
      if (next.x !== camera.x) { camera=next; onCamera(next); return; }
      if (focusOnNextPaint) { focusOnNextPaint=false; revealTask(); }
      surface.style.height = `${layout.height}px`;
      const dpr = global.devicePixelRatio || 1;
      canvas.width=Math.round(layout.width*dpr); canvas.height=Math.round(layout.height*dpr);
      canvas.style.width=`${layout.width}px`; canvas.style.height=`${layout.height}px`;
      ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,layout.width,layout.height);
      const fg=color('--foreground'), secondary=color('--foreground-secondary'), muted=color('--foreground-muted'), divider=color('--border-subtle');
      const family=getComputedStyle(host).fontFamily, mono=color('--font-mono')||family;
      ctx.font=`12px ${mono}`; ctx.textBaseline='middle'; ctx.textAlign='left'; ctx.fillStyle=secondary;
      ctx.fillText('BLOCK',layout.labelX,19); ctx.fillText('Phase0',layout.phaseX,19); ctx.fillText('→',layout.phaseX+layout.phaseW+6,19);
      ctx.fillText('→',layout.syncX+15,19);ctx.fillText('Attention Tasks + Output',layout.taskX,19);
      // One shared barrier, spanning all Blocks. Vertical alignment is a logical boundary.
      ctx.fillStyle=color('--surface-3');ctx.fillRect(layout.syncX-6,layout.header-2,12,layout.height-layout.header);
      ctx.save();ctx.translate(layout.syncX,layout.header+(layout.height-layout.header)/2);ctx.rotate(-Math.PI/2);ctx.fillStyle=fg;ctx.font=`11px ${mono}`;ctx.textAlign='center';ctx.fillText('SYNCALL · 全部 Blocks',0,0);ctx.restore();
      model.lanes.forEach(lane => {
        const y=layout.header+lane.block*layout.row, related=lane.block===selection?.block;
        if(related){ctx.fillStyle=color('--state-selected');ctx.fillRect(0,y,layout.width,layout.row);}
        ctx.strokeStyle=divider;ctx.beginPath();ctx.moveTo(0,y+layout.row);ctx.lineTo(layout.width,y+layout.row);ctx.stroke();
        ctx.fillStyle=related?fg:secondary;ctx.font=`12px ${mono}`;if (!transitionMasked) ctx.fillText(String(lane.block).padStart(2,'0'),layout.labelX+8,y+layout.row/2);
        // Equal-size AIV strips encode work ownership, not Phase0 execution time.
        lane.phase0Lanes.forEach((aiv,index)=>{
          const y0=y+3+index*(layout.row-6)/lane.phase0Lanes.length, h=(layout.row-8)/lane.phase0Lanes.length;
          ctx.fillStyle=aiv.groups.length?colors.phase0:color('--surface-3');ctx.globalAlpha=aiv.groups.length?.65:.5;
          ctx.fillRect(layout.phaseX,y0,layout.phaseW,h);ctx.globalAlpha=1;
          aiv.groups.forEach((group,i)=>{const w=(layout.phaseW-8)/aiv.groups.length;ctx.fillStyle=fg;ctx.globalAlpha=.25;ctx.fillRect(layout.phaseX+4+i*w,y0+2,Math.max(1,w-3),Math.max(1,h-4));ctx.globalAlpha=1;});
        });
        if(!lane.phase0Groups){ctx.fillStyle=muted;ctx.font=`10px ${family}`;ctx.fillText('无有效预处理',layout.phaseX+12,y+layout.row/2);}
        ctx.strokeStyle=divider;ctx.beginPath();ctx.moveTo(layout.phaseX+layout.phaseW+2,y+layout.row/2);ctx.lineTo(layout.syncX-6,y+layout.row/2);ctx.moveTo(layout.syncX+6,y+layout.row/2);ctx.lineTo(layout.taskX-5,y+layout.row/2);ctx.stroke();
        const button=targets.querySelector(`[data-bem-block="${lane.block}"]`);place(button,0,y,layout.taskX-14,layout.row);button.setAttribute('aria-pressed',String(related));
      });
      ctx.save();ctx.beginPath();ctx.rect(layout.taskX,layout.header,Math.max(0,layout.width-layout.taskX),layout.height-layout.header);ctx.clip();
      positions().forEach(rect=>{
        const task=rect.task, selected=selection?.kind==='task'&&selection.task===task.id, related=selection?.block===task.block;
        if (!transitionMasked && transitionFocusBlock !== task.block && semanticTaskMask !== task.id) global.PtoVisualization2TaskEncoding.draw(ctx,{task,...rect,
          colors:{track:color('--surface-4'),fill:color('--primary'),page:color('--surface-2'),text:fg,label:color('--surface-2')},
          selected,related,hovered:hover===task.id,fontFamily:mono});
        // One Output cap belongs to this Task. No synthesized IN/OUT timing segments.
        if(semanticTaskMask !== task.id) pattern.drawTaskBar(ctx,{task:{label:'N↓'},x:rect.outputX,y:rect.y,width:rect.outputWidth,height:rect.height,baseColor:colors.output,isSelected:selected,isRelated:related,fontFamily:mono});
        const button=targets.querySelector(`[data-bem-task="${task.id}"]`), left=Math.max(layout.taskX,rect.x), right=Math.min(layout.width,rect.outputX+rect.outputWidth);
        button.hidden=right<=left;place(button,left,rect.y,Math.max(0,right-left),rect.height);button.setAttribute('aria-pressed',String(selected));
        button.dataset.batch=task.batch;button.dataset.seqLen=task.seqLen;button.dataset.pageCount=task.pages;button.dataset.stackCount=task.stacks;button.dataset.attentionWidth=String(rect.width);button.dataset.outputX=String(rect.outputX);button.dataset.attentionX=String(rect.x);
      });
      ctx.restore();
      host.querySelector('[data-bem-scale]').textContent=`${Math.round(camera.scale*100)}%`;
    }
    function updateInspector() {
      inspector.hidden=!selection||selection.kind==='task';
      if(inspector.hidden)return;
      const lane=model.lanes[selection.block];
      if(!lane){inspector.hidden=true;return;}
      host.querySelector('[data-bem-title]').textContent=`Block ${String(lane.block).padStart(2,'0')}`;
      detail.innerHTML=`<section><h3>Phase0</h3><p>${lane.phase0Groups?`${lane.phase0Lanes.length} 条 AIV Lane · ${lane.phase0Groups} 组预处理`:'无有效预处理工作'}<br>参与 SYNCALL 后进入 Attention。</p>${selection.kind==='lane'?`<p>保留 Assignment 焦点：AIV Lane ${selection.lane}</p>`:''}</section><section><h3>Attention · stride ${model.blocks}</h3><p>${lane.tasks.length} Tasks · ${lane.totalStacks} Stacks</p><div class="viz2-pipeline-task-list">${lane.tasks.map(t=>`<button class="btn btn-ghost btn-sm" type="button" data-bem-select="${t.id}">Task ${t.id}<span>${t.stacks} Stacks</span></button>`).join('')}</div></section><section><h3>Output</h3><p>每个 Task 末尾独立 Normalize 并以 BF16 写回；图中的 N↓ 为等宽符号。</p></section>`;
    }
    function selectTask(id) {const task=model.tasks[id];if(task)onSelect({kind:'task',task:id,block:task.block,batch:task.batch,kvHead:task.kvHead});}
    function update(nextSelection,nextCamera,options={}) {
      selection=nextSelection;camera=nextCamera;active=!!options.active;
      const key=JSON.stringify(selection);
      if(key!==selectionKey){selectionKey=key;updateInspector();focusOnNextPaint=true;}
      if(options.enter){focusOnNextPaint=true;pattern.hideTooltip(tooltip);}
      paint();
    }
    function zoom(factor,pointX=viewport.clientWidth/2){
      if(!layout)return;const c=camera||{scale:1,x:0},scale=Math.max(.5,Math.min(8,c.scale*factor));
      onCamera(bounded({scale,x:pointX-layout.taskX-(pointX-layout.taskX-c.x)*scale/c.scale}));
    }
    host.addEventListener('click',event=>{
      if(event.target.closest('[data-bem-clear]')){onSelect(null);return;}
      if(event.target.closest('[data-bem-fit]')){onCamera({scale:1,x:0});return;}
      const zoomButton=event.target.closest('[data-bem-zoom]');if(zoomButton){zoom(Number(zoomButton.dataset.bemZoom));return;}
      if(suppressClick){suppressClick=false;return;}
      const task=event.target.closest('[data-bem-task], [data-bem-select]');
      if(task){const id=Number(task.dataset.bemTask??task.dataset.bemSelect);if(event.detail===1)doubleCandidate={id,time:event.timeStamp};selectTask(id);return;}
      const block=event.target.closest('[data-bem-block]');if(block)onSelect({kind:'block',block:Number(block.dataset.bemBlock),task:null});
    });
    viewport.addEventListener('dblclick',event=>{const target=event.target.closest('[data-bem-task]');if(target){pattern.hideTooltip(tooltip);onEnter?.(doubleCandidate&&event.timeStamp-doubleCandidate.time<650?doubleCandidate.id:Number(target.dataset.bemTask));doubleCandidate=null;}});
    viewport.addEventListener('wheel',event=>{
      event.preventDefault();const r=viewport.getBoundingClientRect(),unit=event.deltaMode===1?16:event.deltaMode===2?r.height:1;
      pattern.hideTooltip(tooltip);
      if(Math.abs(event.deltaX)>Math.abs(event.deltaY))zoom(Math.exp(Math.max(-.22,Math.min(.22,event.deltaX*unit*.003))),event.clientX-r.left);
      else viewport.scrollTop+=event.deltaY*unit;
    },{passive:false});
    viewport.addEventListener('pointerdown',event=>{if(event.button!==0)return;suppressClick=false;drag={id:event.pointerId,x:event.clientX,startX:event.clientX,startY:event.clientY,moved:false};});
    viewport.addEventListener('pointermove',event=>{
      if(drag){if(!drag.moved&&Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)<=4)return;drag.moved=true;viewport.setPointerCapture(drag.id);viewport.classList.add('is-dragging');pattern.hideTooltip(tooltip);const c=camera||{scale:1,x:0};onCamera(bounded({...c,x:c.x+event.clientX-drag.x}));drag.x=event.clientX;return;}
      const target=event.target.closest('[data-bem-task]'),next=target?Number(target.dataset.bemTask):null;if(hover!==next){hover=next;paint();}
    });
    const endDrag=()=>{if(drag?.moved)suppressClick=true;drag=null;viewport.classList.remove('is-dragging');};
    ['pointerup','pointercancel','lostpointercapture'].forEach(type=>viewport.addEventListener(type,endDrag));
    viewport.addEventListener('pointerleave',()=>{hover=null;pattern.hideTooltip(tooltip);paint();});
    viewport.addEventListener('keydown',event=>{
      if(event.key==='Escape'){onSelect(null);viewport.focus();return;}
      if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;
      event.preventDefault();
      if(selection?.kind==='task'){
        const delta={ArrowLeft:-model.blocks,ArrowRight:model.blocks,ArrowUp:-1,ArrowDown:1}[event.key],next=Math.max(0,Math.min(model.tasks.length-1,selection.task+delta));selectTask(next);
      }else{const block=Math.max(0,Math.min(model.blocks-1,(selection?.block??0)+(event.key==='ArrowUp'?-1:event.key==='ArrowDown'?1:0)));if(event.key==='ArrowRight'&&model.lanes[block].tasks.length)selectTask(model.lanes[block].tasks[0].id);else onSelect({kind:'block',block,task:null});}
    });
    new ResizeObserver(()=>{if(active){focusOnNextPaint=true;paint();}}).observe(viewport);
    function getTransitionSnapshot() {
      if (!layout || !active) return null;
      const r=canvas.getBoundingClientRect(), v=viewport.getBoundingClientRect();
      const rect=(x,y,width,height)=>({x:r.left+x,y:r.top+y,width,height});
      const mono=color('--font-mono')||getComputedStyle(host).fontFamily;
      return {clip:{x:v.left+layout.taskX,y:v.top+layout.header,width:Math.max(0,v.width-layout.taskX),height:Math.max(0,v.height-layout.header)},
        blocks:model.lanes.map(lane=>({id:lane.block,rect:rect(layout.labelX,layout.header+lane.block*layout.row,30,layout.row)})),
        tasks:positions().map(p=>{
          const size=Math.min(11,Math.max(7,(p.width-3)/String(p.task.id).length/.62));
          ctx.font=`${size}px ${mono}`;
          return {id:p.task.id,batch:p.task.batch,task:p.task,rect:rect(p.x,p.y,p.width,p.height),
            output:rect(p.outputX,p.y,p.outputWidth,p.height),label:{size,width:ctx.measureText(String(p.task.id)).width,fontFamily:mono}};
        })};
    }
    function ensureTransitionBlockVisible(block) {
      if (!layout || !active || !model.lanes[block]?.tasks.length) return false;
      const top = layout.header + block * layout.row, bottom = top + layout.row;
      if (top < viewport.scrollTop + layout.header || bottom > viewport.scrollTop + viewport.clientHeight)
        viewport.scrollTop = Math.max(0, top - Math.max(layout.header, (viewport.clientHeight - layout.row) / 2));
      return true;
    }
    return {update,getTransitionSnapshot,ensureTransitionBlockVisible,
      setTransitionFocusBlock(block){transitionFocusBlock=block;paint();},
      setSemanticTaskMask(id){semanticTaskMask=id;paint();},setTransitionMasked(value){transitionMasked=value;paint();},hideTooltip:()=>pattern.hideTooltip(tooltip),model,destroy:()=>tooltipController.destroy()};
  }
  global.PtoVisualization2Pipeline=Object.freeze({createModel,mount});
})(window);
