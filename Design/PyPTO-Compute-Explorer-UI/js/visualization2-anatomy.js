/* Assignment data anatomy. Ownership, logical capacity and modulo resource
 * mapping only: no execution clock or buffer occupancy is inferred. */
(function(global){
  'use strict';
  function createModel({seqLens,blocks,numHeads,kvHeads,headDim,rowTile,pageTokens,pagesPerStack,transferSlots,blockTable=null}){
    const qPerKv=numHeads/kvHeads,stackTokens=pageTokens*pagesPerStack;
    const tasks=seqLens.flatMap((seqLen,batch)=>Array.from({length:kvHeads},(_,kvHead)=>{
      const id=batch*kvHeads+kvHead,block=id%blocks,qStart=kvHead*qPerKv;
      const pages=Array.from({length:Math.ceil(seqLen/pageTokens)},(_,index)=>({index,stack:Math.floor(index/pagesPerStack),
        tokens:Math.min(pageTokens,seqLen-index*pageTokens),physicalId:Number.isInteger(blockTable?.[batch]?.[index])&&blockTable[batch][index]>=0?blockTable[batch][index]:null}));
      const stacks=global.PtoVisualization2TaskEncoding.segments(seqLen,stackTokens,pageTokens).map(segment=>({
        index:segment.index,tokens:segment.tokens,slot:segment.index%transferSlots,
        pages:pages.filter(page=>page.stack===segment.index).map(page=>page.index),capacityPages:segment.pages}));
      return {id,block,batch,kvHead,seqLen,pageTokens,pagesPerStack,stackTokens,transferSlots,qHeads:Array.from({length:qPerKv},(_,q)=>qStart+q),
        q:{shape:[rowTile,headDim],validShape:[qPerKv,headDim]},pages,stacks,
        slots:Array.from({length:transferSlots},(_,index)=>({index,stacks:stacks.filter(stack=>stack.slot===index).map(stack=>stack.index)})),
        output:{shape:[qPerKv,headDim],dtype:'BF16',rowStart:batch*numHeads+qStart,rowEnd:batch*numHeads+qStart+qPerKv-1}};
    }));
    return {tasks,blocks,numHeads,kvHeads,headDim,rowTile,pageTokens,pagesPerStack,stackTokens,transferSlots};
  }
  function related(task,focus){
    let stacks=[];
    if(focus?.kind==='stack')stacks=[focus.index];
    if(focus?.kind==='page')stacks=[task.pages[focus.index]?.stack];
    if(focus?.kind==='slot')stacks=task.slots[focus.index]?.stacks||[];
    stacks=stacks.filter(index=>task.stacks[index]);
    const slots=[...new Set(stacks.map(index=>task.stacks[index].slot))];
    if(focus?.kind==='slot'&&!slots.includes(focus.index))slots.push(focus.index);
    return {stacks,slots,pages:stacks.flatMap(index=>task.stacks[index].pages),
      shared:task.stacks.filter(stack=>slots.includes(stack.slot)).map(stack=>stack.index)};
  }
  // Browser wheel events have no portable gesture ID. Keep recognition independent
  // from animation, with one pending command rather than a cooldown/scroll lock.
  function createPagingInput({isAnimating,onPage}){
    const config={threshold:30,noise:2,quiet:180,notchGap:35,restartGap:45,restartGain:1.8,decay:.55};
    let phase='Ready',direction=0,sum=0,lastTime=-Infinity,lastMagnitude=0,peak=0,trough=Infinity,reverseSum=0,pending=0;
    function resetGesture(){phase='Ready';direction=0;sum=0;peak=0;trough=Infinity;reverseSum=0;}
    function request(sign){if(isAnimating())pending=sign;else onPage(sign);}
    function input({delta,time,discrete=false}){
      const magnitude=Math.abs(delta),sign=Math.sign(delta),gap=time-lastTime;
      if(magnitude<config.noise)return;
      if(gap>=config.quiet)resetGesture();
      if(phase!=='Ready'){
        reverseSum=sign!==direction?reverseSum+magnitude:0;
        const reversal=reverseSum>=config.threshold;
        const notch=discrete&&gap>=config.notchGap;
        const acceleration=magnitude>=Math.max(12,trough*config.restartGain);
        const strongImpulse=magnitude>=Math.max(30,trough*3,peak*.65);
        const restart=sign===direction&&phase==='Momentum'&&trough<=peak*config.decay&&
          acceleration&&(gap>=config.restartGap||strongImpulse);
        if(reversal||notch||restart)resetGesture();
        else if(magnitude<lastMagnitude*.85||magnitude<=peak*config.decay)phase='Momentum';
      }
      lastTime=time;lastMagnitude=magnitude;
      if(phase==='Ready'){
        if(sign!==direction){direction=sign;sum=0;}
        sum+=magnitude;peak=Math.max(peak,magnitude);
        if(sum>=config.threshold||discrete){phase='Triggered';trough=magnitude;request(sign);}
      }else{peak=Math.max(peak,magnitude);trough=Math.min(trough,magnitude);}
    }
    return {input,flush(){const sign=pending;pending=0;if(sign)onPage(sign);},
      reset(){resetGesture();lastTime=-Infinity;lastMagnitude=0;pending=0;},
      getState(){return {phase,pending};}};
  }
  function mount(host,{model,scenario,onTask,onFocus,onCamera,onInspector,onExit}){
    const esc=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    const listeners=new AbortController(),listen=(el,type,fn,options={})=>el.addEventListener(type,fn,{...options,signal:listeners.signal});
    host.classList.add('viz2-anatomy','viz2-compute','viz2-assignment-detail');
    host.innerHTML=`${global.PtoComputeDetailControls.identity()}<div class="compute-parameter-control compute-sequence-control" data-compute-toolbar></div><button class="btn btn-ghost btn-sm" data-anatomy-back>← 返回 Block 间</button>
      <div class="viz2-anatomy-body"><div class="viz2-anatomy-stage"><div class="viz2-anatomy-workspace">
        <div class="viz2-anatomy-viewport" role="region" tabindex="0" aria-label="当前 Task KV Cache，纵向浏览 Stacks"><div class="viz2-anatomy-track" data-anatomy-track></div></div>
        <section class="viz2-anatomy-query" aria-label="固定 Q Tile 结构模板"><div data-anatomy-query></div><p>结构模板 · 各 Batch 使用各自的 Q 数据</p></section>
        <div class="kf-pa-map__tooltip" role="tooltip" hidden></div></div>
        <footer class="viz2-anatomy-footer"><span>有效数据 <i class="viz2-anatomy-key"></i>未用容量 <i class="viz2-anatomy-key is-capacity"></i><small>纵向浏览 Stacks</small></span><div class="kf-pa-map__tools"><button class="btn btn-ghost btn-sm" data-anatomy-zoom="0.8" aria-label="缩小 KV Cache">−</button><output data-anatomy-scale>100%</output><button class="btn btn-ghost btn-sm" data-anatomy-zoom="1.25" aria-label="放大 KV Cache">＋</button><button class="btn btn-ghost btn-sm" data-anatomy-fit>适应</button></div></footer></div>
        <aside class="pto-ide-frame__pane viz2-anatomy-inspector" aria-label="Task 数据详情" hidden><header class="pto-ide-frame__pane-header"><b data-anatomy-title></b><button class="btn btn-ghost btn-sm" data-anatomy-close aria-label="关闭数据详情">×</button></header><div class="pto-ide-frame__pane-body" data-anatomy-detail></div></aside></div>`;
    const controls=global.PtoComputeDetailControls,toolbar=host.querySelector('[data-compute-toolbar]');
    toolbar.innerHTML=controls.sequence();
    host.querySelector('[data-compute-identity]').innerHTML=controls.picker(model,'Assignment Task');
    const syncScenario=controls.connect(host,scenario);
    const input=toolbar.querySelector('[data-compute-pages]'),range=toolbar.querySelector('[data-page-range]');
    input.max=range.max=global.PtoPagedAttentionIntro.model.constants.maxSeq;
    const viewport=host.querySelector('.viz2-anatomy-viewport'),track=host.querySelector('[data-anatomy-track]'),query=host.querySelector('[data-anatomy-query]'),inspector=host.querySelector('.viz2-anatomy-inspector'),tooltip=host.querySelector('[role="tooltip"]');
    let task=null,focus=null,active=false,cards=[],scale=1;
    const rect=(x,y,w,h,attrs='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" ${attrs}/>`;
    const text=(x,y,value,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${esc(value)}</text>`;
    const object=(kind,index,label,body)=>`<g data-anatomy-kind="${kind}" data-anatomy-index="${index}" role="button" tabindex="0" aria-label="${esc(label)}" aria-pressed="false">${body}</g>`;
    function card(t){
      return `<article class="viz2-anatomy-card" data-anatomy-task="${t.id}" role="group" aria-label="${scenario.get().isCustom?'Custom · ':''}KV Cache · ${t.seqLen} tokens"><div class="viz2-anatomy-cache-scroll" tabindex="0" aria-label="Task ${t.id} 的全部 KV Cache Stacks"><svg class="viz2-anatomy-svg" preserveAspectRatio="xMinYMin meet" viewBox="0 0 368 ${Math.max(84,t.stacks.length*104)}" style="height:${Math.max(84,t.stacks.length*104)}px" aria-label="KV Cache · 逻辑 Pages 与有效容量">${t.stacks.map(stack=>{
        const y=stack.index*104;
        return `<g data-anatomy-stack-container="${stack.index}">${object('stack',stack.index,`Stack ${stack.index}，${stack.tokens}/${t.stackTokens} tokens`,rect(0,y,368,96,'class="viz2-anatomy-stack-boundary"')+text(12,y+23,`Stack ${stack.index} · ${stack.tokens}/${t.stackTokens} tokens`))}${Array.from({length:t.pagesPerStack},(_,slot)=>{
          const index=stack.index*t.pagesPerStack+slot,page=t.pages[index],gap=8,w=(344-gap*(t.pagesPerStack-1))/t.pagesPerStack,x=12+slot*(w+gap),key=`${stack.index}:${slot}`;
          if(!page)return rect(x,y+34,w,50,`class="viz2-anatomy-unused-page" data-anatomy-cell="${key}"`);
          return object('page',index,`逻辑 Page ${index}，${page.tokens}/${t.pageTokens} tokens`,rect(x,y+34,w,50,`class="viz2-anatomy-empty" data-anatomy-cell="${key}"`)+rect(x,y+34,w*page.tokens/t.pageTokens,50,`class="viz2-anatomy-valid" data-anatomy-page-fill="${index}"`)+`<text x="${x+7}" y="${y+53}" class="viz2-anatomy-page-label ${page.tokens<t.pageTokens/2?'is-tail':''}"><tspan x="${x+7}">P${index}</tspan><tspan x="${x+7}" dy="18">${page.tokens} tokens</tspan></text>`);
        }).join('')}</g>`;
      }).join('')}</svg></div></article>`;
    }
    function build(){
      const height=task.q.shape[0]*9+20;
      query.innerHTML=`<h3>Q Tile</h3><svg class="viz2-anatomy-svg" viewBox="0 0 190 ${height}" style="height:auto" aria-label="Q Tile 结构模板">${object('q',0,'Q Tile 结构模板',rect(0,0,190,height,'class="viz2-anatomy-q-boundary"')+Array.from({length:task.q.shape[0]},(_,row)=>rect(16,10+row*9,158,7,`class="${row<task.qHeads.length?'viz2-anatomy-valid':'viz2-anatomy-empty'}" ${row<task.qHeads.length?`data-anatomy-q-row="${row}"`:''}`)).join(''))}</svg><b>Q Heads ${task.qHeads[0]}–${task.qHeads.at(-1)}</b><p>Tile Shape · ${task.q.shape.join(' × ')}</p><p>Valid Shape · ${task.q.validShape.join(' × ')}</p>`;
      track.innerHTML=card(task);cards=[...track.children];
    }
    function paint(){
      if(!active||!task)return;
      const svg=track.querySelector('svg');
      svg.style.width=`${368*scale}px`;svg.style.height=`${Math.max(84,task.stacks.length*104)*scale}px`;
      host.querySelector('[data-anatomy-scale]').textContent=`${Math.round(scale*100)}%`;
    }
    function describe(t,kind,index){
      if(kind==='q')return `结构模板 · Q Heads ${t.qHeads[0]}–${t.qHeads.at(-1)} · 各 Batch 的 Q 数值独立`;
      if(kind==='page'){const page=t.pages[index];return `逻辑 Page ${index} · ${page.tokens}/${t.pageTokens} tokens · Stack ${page.stack}${page.physicalId===null?' · 物理 Page 未知':` · 物理 Page ${page.physicalId}`}`;}
      const stack=t.stacks[index];return `Stack ${index} · ${stack.tokens}/${t.stackTokens} tokens · Pages ${stack.pages.join(', ')}`;
    }
    function paintFocus(){
      host.querySelectorAll('[data-anatomy-kind]').forEach(element=>{
        const t=element.closest('[data-anatomy-task]'),local=!t||Number(t.dataset.anatomyTask)===task.id;
        const kind=element.dataset.anatomyKind,index=Number(element.dataset.anatomyIndex),selected=local&&focus?.kind===kind&&focus.index===index,r=related(task,focus);
        element.classList.toggle('is-selected',selected);element.setAttribute('aria-pressed',String(selected));
        element.classList.toggle('is-related',local&&(kind==='stack'?r.stacks.includes(index):kind==='page'?r.pages.includes(index):false));
      });
    }
    function paintInspector(open){
      const valid=focus?.kind==='q'||focus?.kind==='page'&&task.pages[focus.index]||focus?.kind==='stack'&&task.stacks[focus.index];
      inspector.hidden=!open||!valid;if(inspector.hidden)return;
      const {kind,index}=focus;
      host.querySelector('[data-anatomy-title]').textContent=kind==='q'?'Q Tile 结构模板':`${kind==='page'?'逻辑 Page':'Stack'} ${index}`;
      let rows;
      if(kind==='q')rows=[['Q Heads',task.qHeads.join(', ')],['Tile Shape',task.q.shape.join(' × ')],['Valid Shape',task.q.validShape.join(' × ')],['数据语义','共用结构示意；不同 Batch 的 Q Tensor 数值独立']];
      else if(kind==='page'){const p=task.pages[index];rows=[['逻辑 Page Index',index],['有效 Token',`${p.tokens} / ${task.pageTokens}`],['Stack',p.stack],['物理 Page ID',p.physicalId===null?'未知 · 未提供实际 block_table 数据':p.physicalId]];}
      else {const stack=task.stacks[index];rows=[['Stack ID',index],['有效 Token',`${stack.tokens} / ${task.stackTokens}`],['逻辑 Pages',stack.pages.join(', ')],['未用容量',`${task.stackTokens-stack.tokens} tokens`]];}
      host.querySelector('[data-anatomy-detail]').innerHTML=`<p>${scenario.get().isCustom?'Task N · Batch N · KV Head N':`Task ${task.id} · Batch ${task.batch} · KV Head ${task.kvHead}`}</p><dl>${rows.map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl>`;
    }
    function update(id,nextFocus,nextCamera,{active:visible,inspectorOpen=false}={}){
      active=!!visible;if(!active){tooltip.hidden=true;return;}
      const next=scenario.task('assignment',id)||model.tasks[id];if(!next)return;
      const changed=task!==next;task=next;
      focus=nextFocus;
      if((focus?.kind==='page'&&!task.pages[focus.index])||(focus?.kind==='stack'&&!task.stacks[focus.index]))focus=null;
      scale=Math.max(.25,Math.min(2,nextCamera?.scale||1));
      if(changed){build();toolbar.querySelector('[data-page-error]').textContent='';input.removeAttribute('aria-invalid');}
      const s=scenario.get(),picker=host.querySelector('[data-compute-task]');picker.value=id;picker.dataset.displayLabel=s.isCustom?'Task N':`Task ${id}`;
      host.querySelector('[data-compute-owner]').textContent=s.isCustom?'Block N / KV Head N':`Block ${String(task.block).padStart(2,'0')} / KV Head ${task.kvHead}`;
      host.querySelector('[data-compute-facts]').textContent=`Batch ${s.isCustom?'N':task.batch}\n${s.effectiveSeqLen} tokens`;
      input.value=range.value=s.effectiveSeqLen;
      toolbar.querySelector('[data-sequence-pages]').textContent=s.pageCount;
      toolbar.querySelector('[data-sequence-stacks]').textContent=s.stackCount;
      toolbar.querySelector('[data-page-step="-1"]').disabled=s.effectiveSeqLen<=1;
      toolbar.querySelector('[data-page-step="1"]').disabled=s.effectiveSeqLen>=Number(input.max);
      toolbar.querySelector('[data-page-summary]').textContent=s.isCustom?'Custom · 模型推导数据组织':'';
      syncScenario();paint();paintFocus();paintInspector(inspectorOpen);
    }
    function configure(value){const result=scenario.edit('sequence',value);toolbar.querySelector('[data-page-error]').textContent=result.error||'';if(result.error)input.setAttribute('aria-invalid','true');else input.removeAttribute('aria-invalid');}
    listen(input,'input',()=>configure(input.value));listen(range,'input',()=>configure(range.value));
    listen(host.querySelector('[data-compute-task]'),'change',event=>onTask(Number(event.target.value)));
    function zoom(factor){onCamera(task.id,{scale:Math.max(.25,Math.min(2,scale*factor)),x:0,y:0});}
    listen(host,'click',event=>{
      tooltip.hidden=true;
      const step=event.target.closest('[data-page-step]');if(step){configure(task.seqLen+Number(step.dataset.pageStep));return;}
      if(event.target.closest('[data-anatomy-back]')){onExit();return;}
      if(event.target.closest('[data-anatomy-close]')){onInspector(false);return;}
      if(event.target.closest('[data-anatomy-fit]')){
        const scroller=track.querySelector('.viz2-anatomy-cache-scroll');
        onCamera(task.id,{scale:Math.max(.25,Math.min(1,(scroller.clientWidth-8)/368,(scroller.clientHeight-8)/Math.max(84,task.stacks.length*104))),x:0,y:0});scroller.scrollTop=scroller.scrollLeft=0;return;
      }
      const z=event.target.closest('[data-anatomy-zoom]');if(z){zoom(Number(z.dataset.anatomyZoom));return;}
      const object=event.target.closest('[data-anatomy-kind]');if(object)onFocus({kind:object.dataset.anatomyKind,index:Number(object.dataset.anatomyIndex)});
    });
    listen(viewport,'wheel',event=>{if(event.ctrlKey||event.metaKey){event.preventDefault();zoom(Math.exp(-event.deltaY*.002));}},{passive:false});
    listen(host,'pointermove',event=>{
      if(!active)return;const object=event.target.closest('[data-anatomy-kind]');tooltip.hidden=!object;if(!object)return;
      tooltip.textContent=describe(task,object.dataset.anatomyKind,Number(object.dataset.anatomyIndex));
      const r=host.querySelector('.viz2-anatomy-workspace').getBoundingClientRect();tooltip.style.left=`${Math.max(8,Math.min(event.clientX-r.left+14,r.width-tooltip.offsetWidth-8))}px`;tooltip.style.top=`${Math.max(8,Math.min(event.clientY-r.top+16,r.height-tooltip.offsetHeight-8))}px`;
    });
    listen(host,'pointerleave',()=>{tooltip.hidden=true;});
    listen(host,'keydown',event=>{
      if(!active||event.target.closest('input,select,button'))return;
      if(event.key==='Escape')onInspector(false);
      else if(['+','=','-','0'].includes(event.key)){event.preventDefault();if(event.key==='0')host.querySelector('[data-anatomy-fit]').click();else zoom(event.key==='-'?.8:1.25);}
      else if(['Enter',' '].includes(event.key)&&event.target.closest('[data-anatomy-kind]')){event.preventDefault();event.target.closest('[data-anatomy-kind]').dispatchEvent(new MouseEvent('click',{bubbles:true}));}
    });
    const resizeObserver=new ResizeObserver(()=>{if(active)paint();});resizeObserver.observe(viewport);
    const screenRect=element=>{const r=element.getBoundingClientRect();return {x:r.left,y:r.top,width:r.width,height:r.height};};
    function getTransitionSnapshot(){
      if(!task||!active||!cards[0])return null;const element=cards[0];
      return {task:task.id,model:task,frame:screenRect(element),title:screenRect(host.querySelector('[data-compute-identity]')),qRows:[...query.querySelectorAll('[data-anatomy-q-row]')].map(screenRect),cells:[...element.querySelectorAll('[data-anatomy-cell]')].map(cell=>({key:cell.dataset.anatomyCell,rect:screenRect(cell)}))};
    }
    return {update,model,getTransitionSnapshot,hideTooltip(){tooltip.hidden=true;},destroy(){active=false;resizeObserver.disconnect();listeners.abort();host.replaceChildren();}};
  }
  global.PtoVisualization2Anatomy=Object.freeze({createModel,related,mount,createPagingInput});
})(window);
