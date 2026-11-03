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
  function mount(host,{model,onTask,onFocus,onCamera,onInspector,onExit}){
    const esc=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    const media=global.matchMedia('(prefers-reduced-motion: reduce)');
    host.classList.add('viz2-anatomy');
    host.innerHTML=`<div class="viz2-anatomy-taskbar"><h2 data-anatomy-block></h2><span data-anatomy-kv></span><nav class="viz2-anatomy-task-nav" aria-label="Block 内 Task 快捷导航" data-anatomy-task-nav></nav><button class="btn btn-ghost btn-sm" data-anatomy-back>← 返回 Block 间</button></div>
      <div class="viz2-anatomy-body"><div class="viz2-anatomy-stage"><div class="viz2-anatomy-workspace">
        <section class="viz2-anatomy-query" aria-label="固定 Q Tile 结构模板"><div data-anatomy-query></div><p>结构模板 · 各 Batch 使用各自的 Q 数据</p></section>
        <div class="viz2-anatomy-viewport" id="viz2-anatomy-carousel" role="region" aria-roledescription="轮播" tabindex="0" aria-label="Block 内 Task KV Cache，左右方向键切换，横向拖动或滚动浏览"><div class="viz2-anatomy-track" data-anatomy-track></div></div>
        <div class="kf-pa-map__tooltip" role="tooltip" hidden></div></div>
        <footer class="viz2-anatomy-footer"><span>有效数据 <i class="viz2-anatomy-key"></i>未用容量 <i class="viz2-anatomy-key is-capacity"></i><small>横向浏览 Task · 卡片内纵向浏览 Stacks</small></span><div class="kf-pa-map__tools"><button class="btn btn-ghost btn-sm" data-anatomy-prev aria-label="上一个 Task">←</button><output data-anatomy-position></output><button class="btn btn-ghost btn-sm" data-anatomy-next aria-label="下一个 Task">→</button><button class="btn btn-ghost btn-sm" data-anatomy-zoom="0.8" aria-label="缩小 KV Cache">−</button><output data-anatomy-scale>100%</output><button class="btn btn-ghost btn-sm" data-anatomy-zoom="1.25" aria-label="放大 KV Cache">＋</button><button class="btn btn-ghost btn-sm" data-anatomy-fit>适应</button></div></footer></div>
        <aside class="pto-ide-frame__pane viz2-anatomy-inspector" aria-label="Task 数据详情" hidden><header class="pto-ide-frame__pane-header"><b data-anatomy-title></b><button class="btn btn-ghost btn-sm" data-anatomy-close aria-label="关闭数据详情">×</button></header><div class="pto-ide-frame__pane-body" data-anatomy-detail></div></aside></div>`;
    const viewport=host.querySelector('.viz2-anatomy-viewport'),track=host.querySelector('[data-anatomy-track]'),query=host.querySelector('[data-anatomy-query]'),inspector=host.querySelector('aside'),tooltip=host.querySelector('[role="tooltip"]');
    let task=null,focus=null,active=false,block=null,tasks=[],cards=[],position=0,target=0,velocity=0,frame=0,drag=null,suppressClick=false,scale=1,pitch=440,pendingFocus=null,navButtons=[],navTask=null;
    const paging=createPagingInput({isAnimating:()=>!!frame,onPage:direction=>{
      const next=clamp(target+direction);
      if(next!==target){pendingFocus=null;tooltip.hidden=true;snap(next,true);}
    }});
    const clamp=index=>Math.max(0,Math.min(tasks.length-1,index));
    const rect=(x,y,w,h,attrs='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" ${attrs}/>`;
    const text=(x,y,value,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${esc(value)}</text>`;
    const object=(kind,index,label,body)=>`<g data-anatomy-kind="${kind}" data-anatomy-index="${index}" role="button" tabindex="0" aria-label="${esc(label)}" aria-pressed="false">${body}</g>`;
    function card(t){
      return `<article class="viz2-anatomy-card" data-anatomy-task="${t.id}" role="group" aria-roledescription="幻灯片" aria-label="Task ${t.id}，Batch ${t.batch}，${t.seqLen} tokens"><header><h3 class="viz2-anatomy-task-title">Task ${t.id}</h3><p>Batch ${t.batch} · ${t.seqLen} tokens</p><p>${t.stacks.length} Stacks · ${t.pages.length} Pages · 128 tokens / Page</p></header><div class="viz2-anatomy-cache-scroll" tabindex="0" aria-label="Task ${t.id} 的全部 KV Cache Stacks"><svg class="viz2-anatomy-svg" preserveAspectRatio="xMinYMin meet" viewBox="0 0 368 ${Math.max(84,t.stacks.length*104)}" style="height:${Math.max(84,t.stacks.length*104)}px" aria-label="KV Cache · 逻辑 Pages 与有效容量">${t.stacks.map(stack=>{
        const y=stack.index*104;
        return `<g data-anatomy-stack-container="${stack.index}">${object('stack',stack.index,`Stack ${stack.index}，${stack.tokens}/${t.stackTokens} tokens`,rect(0,y,368,96,'class="viz2-anatomy-stack-boundary"')+text(12,y+23,`Stack ${stack.index} · ${stack.tokens}/${t.stackTokens} tokens`))}${Array.from({length:t.pagesPerStack},(_,slot)=>{
          const index=stack.index*t.pagesPerStack+slot,page=t.pages[index],gap=8,w=(344-gap*(t.pagesPerStack-1))/t.pagesPerStack,x=12+slot*(w+gap),key=`${stack.index}:${slot}`;
          if(!page)return rect(x,y+34,w,50,`class="viz2-anatomy-unused-page" data-anatomy-cell="${key}"`);
          return object('page',index,`逻辑 Page ${index}，${page.tokens}/${t.pageTokens} tokens`,rect(x,y+34,w,50,`class="viz2-anatomy-empty" data-anatomy-cell="${key}"`)+rect(x,y+34,w*page.tokens/t.pageTokens,50,`class="viz2-anatomy-valid" data-anatomy-page-fill="${index}"`)+`<text x="${x+7}" y="${y+53}" class="viz2-anatomy-page-label ${page.tokens<t.pageTokens/2?'is-tail':''}"><tspan x="${x+7}">P${index}</tspan><tspan x="${x+7}" dy="18">${page.tokens} tokens</tspan></text>`);
        }).join('')}</g>`;
      }).join('')}</svg></div></article>`;
    }
    function buildBlock(){
      block=task.block;tasks=model.tasks.filter(t=>t.block===block).sort((a,b)=>a.id-b.id);
      host.querySelector('[data-anatomy-block]').textContent=`Block ${String(block).padStart(2,'0')}`;
      host.querySelector('[data-anatomy-kv]').textContent=`KV Head ${[...new Set(tasks.map(t=>t.kvHead))].join(', ')}`;
      const navigation=host.querySelector('[data-anatomy-task-nav]');
      navigation.innerHTML=tasks.map(t=>`<button type="button" class="btn btn-ghost btn-sm" data-anatomy-task-go="${t.id}" aria-current="false" aria-controls="viz2-anatomy-carousel">Task ${t.id}</button>`).join('');
      navButtons=[...navigation.children];navTask=null;
      const height=task.q.shape[0]*9+20;
      query.innerHTML=`<h3>Q Tile</h3><svg class="viz2-anatomy-svg" viewBox="0 0 190 ${height}" style="height:${height}px" aria-label="Q Tile 结构模板">${object('q',0,'Q Tile 结构模板',rect(0,0,190,height,'class="viz2-anatomy-q-boundary"')+Array.from({length:task.q.shape[0]},(_,row)=>rect(16,10+row*9,158,7,`class="${row<task.qHeads.length?'viz2-anatomy-valid':'viz2-anatomy-empty'}" ${row<task.qHeads.length?`data-anatomy-q-row="${row}"`:''}`)).join(''))}</svg><b>Q Heads ${task.qHeads[0]}–${task.qHeads.at(-1)}</b><p>Tile Shape · ${task.q.shape.join(' × ')}</p><p>Valid Shape · ${task.q.validShape.join(' × ')}</p>`;
      track.innerHTML=tasks.map(card).join('');cards=[...track.children];
      position=target=tasks.findIndex(t=>t.id===task.id);velocity=0;
    }
    function paint(){
      if(!active||!tasks.length)return;
      const width=Math.max(200,Math.min(520,viewport.clientWidth*.68)),gap=24;pitch=width+gap;
      track.style.transform=`translate3d(${viewport.clientWidth/2-width/2-position*pitch}px,0,0)`;
      cards.forEach((element,index)=>{
        element.style.width=`${width}px`;element.style.left=`${index*pitch}px`;
        const distance=Math.abs(index-position);element.style.opacity=String(Math.max(.25,1-distance*.48));
        element.classList.toggle('is-current',element.dataset.anatomyTask===String(task.id));
        element.classList.toggle('is-before',index<position-.5);
        element.setAttribute('aria-current',String(element.dataset.anatomyTask===String(task.id)));
        const svg=element.querySelector('svg');svg.style.width=`${scale*100}%`;svg.style.height=`${Math.max(84,tasks[index].stacks.length*84)*scale}px`;
      });
      if(navTask!==task.id){
        navTask=task.id;
        navButtons.forEach(button=>{const selected=Number(button.dataset.anatomyTaskGo)===task.id;button.classList.toggle('is-selected',selected);button.setAttribute('aria-current',String(selected));});
        navButtons.find(button=>Number(button.dataset.anatomyTaskGo)===task.id)?.scrollIntoView({block:'nearest',inline:'nearest'});
      }
      host.querySelector('[data-anatomy-position]').textContent=`${tasks.findIndex(t=>t.id===task.id)+1} / ${tasks.length}`;
      host.querySelector('[data-anatomy-prev]').disabled=target<=0;
      host.querySelector('[data-anatomy-next]').disabled=target>=tasks.length-1;
      host.querySelector('[data-anatomy-scale]').textContent=`${Math.round(scale*100)}%`;
    }
    function stop(){cancelAnimationFrame(frame);frame=0;paging.reset();drag=null;pendingFocus=null;tooltip.hidden=true;}
    function commit(){
      const next=tasks[target];if(!next)return;
      if(next.id!==task.id)onTask(next.id);
      if(pendingFocus){const nextFocus=pendingFocus;pendingFocus=null;onFocus(nextFocus);}
    }
    function snap(index,fromWheel=false){
      if(!fromWheel)paging.reset();
      cancelAnimationFrame(frame);target=clamp(Math.round(index));velocity=0;
      const origin=position,duration=400;let started=null;
      if(media.matches){position=target;frame=0;paint();commit();paging.flush();return;}
      function tick(now){
        if(!active){frame=0;return;}
        if(started===null)started=now;
        const progress=Math.min(1,(now-started)/duration);
        // A bounded full-page move, with zero velocity at both endpoints.
        const eased=progress*progress*progress*(progress*(progress*6-15)+10);
        position=origin+(target-origin)*eased;paint();
        if(progress===1){position=target;frame=0;paint();commit();paging.flush();}
        else frame=requestAnimationFrame(tick);
      }
      frame=requestAnimationFrame(tick);
    }
    function resistance(value){const bound=clamp(value);return bound+(value-bound)*.22;}
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
      host.querySelector('[data-anatomy-detail]').innerHTML=`<p>Task ${task.id} · Batch ${task.batch} · KV Head ${task.kvHead}</p><dl>${rows.map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl>`;
    }
    function update(id,nextFocus,nextCamera,{active:visible,inspectorOpen=false}={}){
      const entering=visible&&!active;active=visible;
      if(!active){stop();return;}
      const next=model.tasks[id];if(!next)return;
      const changed=task?.id!==id;task=next;focus=nextFocus;
      scale=Math.max(.6,Math.min(2,nextCamera?.scale||1));
      if(block!==task.block){stop();buildBlock();}
      else if(entering){stop();position=target=tasks.findIndex(t=>t.id===id);velocity=0;}
      else if(changed){const index=tasks.findIndex(t=>t.id===id);if(Math.abs(position-index)>.001)snap(index);else target=index;}
      paint();paintFocus();paintInspector(inspectorOpen);
    }
    function zoom(factor){const next=Math.max(.6,Math.min(2,scale*factor));if(factor<1&&scale<=.6){onExit();return;}onCamera(task.id,{scale:next,x:0,y:0});}
    host.addEventListener('click',event=>{
      if(suppressClick){suppressClick=false;event.preventDefault();return;}
      tooltip.hidden=true;
      const taskButton=event.target.closest('[data-anatomy-task-go]');
      if(taskButton){const id=Number(taskButton.dataset.anatomyTaskGo);if(id!==task.id){pendingFocus=null;snap(tasks.findIndex(t=>t.id===id));}return;}
      if(event.target.closest('[data-anatomy-back]')){stop();onExit();return;}
      if(event.target.closest('[data-anatomy-close]')){onInspector(false);return;}
      if(event.target.closest('[data-anatomy-fit]')){onCamera(task.id,{scale:1,x:0,y:0});snap(tasks.findIndex(t=>t.id===task.id),0);return;}
      const zoomButton=event.target.closest('[data-anatomy-zoom]');if(zoomButton){zoom(Number(zoomButton.dataset.anatomyZoom));return;}
      if(event.target.closest('[data-anatomy-prev]')){snap(target-1);return;}
      if(event.target.closest('[data-anatomy-next]')){snap(target+1);return;}
      const card=event.target.closest('[data-anatomy-task]'),object=event.target.closest('[data-anatomy-kind]');
      const nextFocus=object?{kind:object.dataset.anatomyKind,index:Number(object.dataset.anatomyIndex)}:null;
      if(card&&Number(card.dataset.anatomyTask)!==task.id){pendingFocus=nextFocus;snap(tasks.findIndex(t=>t.id===Number(card.dataset.anatomyTask)));return;}
      if(nextFocus)onFocus(nextFocus);
    });
    viewport.addEventListener('wheel',event=>{
      const unit=event.deltaMode===1?16:event.deltaMode===2?viewport.clientWidth:1;
      if(event.ctrlKey||event.metaKey){event.preventDefault();zoom(Math.exp(-event.deltaY*unit*.002));return;}
      // Ordinary vertical wheel input belongs to the card's Stack scroller.
      // Shift+wheel is the explicit horizontal paging gesture on wheel mice.
      const horizontal=event.shiftKey&&event.deltaX===0?event.deltaY:event.deltaX;
      if(!horizontal||(!event.shiftKey&&Math.abs(horizontal)<=Math.abs(event.deltaY)))return;
      event.preventDefault();
      if(drag){
        if(event.buttons&1)return;
        endDrag(true);
      }
      paging.input({delta:horizontal*unit,time:performance.now(),
        discrete:event.deltaMode===1||event.deltaMode===2||
          (event.shiftKey&&event.deltaX===0&&Number.isInteger(event.deltaY)&&Math.abs(event.deltaY)>=40)});
    },{passive:false});
    viewport.addEventListener('pointerdown',event=>{
      if(event.button!==0||event.target.closest('button'))return;
      stop();pendingFocus=null;suppressClick=false;velocity=0;
      drag={id:event.pointerId,x:event.clientX,y:event.clientY,lastX:event.clientX,lastTime:performance.now(),position,index:clamp(Math.round(position)),moved:false};
    });
    viewport.addEventListener('pointermove',event=>{
      if(!drag)return;
      if(!(event.buttons&1)){endDrag(true);return;}
      const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
      if(!drag.moved){if(Math.abs(dy)>Math.abs(dx)&&Math.abs(dy)>6){drag=null;return;}if(Math.abs(dx)<5)return;drag.moved=true;viewport.setPointerCapture(drag.id);}
      event.preventDefault();const now=performance.now(),dt=Math.max(8,now-drag.lastTime);
      velocity=.65*velocity+.35*(drag.lastX-event.clientX)/pitch/dt*1000;
      const offset=-dx/pitch;
      const bounded=Math.sign(offset)*Math.min(.92,Math.abs(offset));
      position=resistance(drag.index+bounded);drag.lastX=event.clientX;drag.lastTime=now;paint();
    });
    function endDrag(cancelled=false){if(!drag)return;const current=drag;drag=null;suppressClick=current.moved;
      if(current.moved){
        if(performance.now()-current.lastTime>100)velocity=0;
        const offset=position-current.index;
        const direction=Math.abs(offset)>.18?Math.sign(offset):Math.abs(velocity)>.55?Math.sign(velocity):0;
        snap(current.index+(cancelled?0:direction));
      }else if(Math.abs(position-current.index)>.001)snap(current.index);
    }

    const releasePointer=()=>endDrag();
    window.addEventListener('pointerup',releasePointer);
    viewport.addEventListener('pointercancel',()=>endDrag(true));viewport.addEventListener('lostpointercapture',()=>endDrag(true));
    host.addEventListener('pointermove',event=>{
      if(drag||frame){tooltip.hidden=true;return;}
      const object=event.target.closest('[data-anatomy-kind]');tooltip.hidden=!object;if(!object)return;
      const id=object.closest('[data-anatomy-task]')?.dataset.anatomyTask,t=id==null?task:model.tasks[Number(id)];if(!t)return;
      tooltip.textContent=describe(t,object.dataset.anatomyKind,Number(object.dataset.anatomyIndex));
      const r=host.querySelector('.viz2-anatomy-workspace').getBoundingClientRect();tooltip.style.left=`${Math.max(8,Math.min(event.clientX-r.left+14,r.width-tooltip.offsetWidth-8))}px`;tooltip.style.top=`${Math.max(8,Math.min(event.clientY-r.top+16,r.height-tooltip.offsetHeight-8))}px`;
    });
    host.addEventListener('pointerleave',()=>{tooltip.hidden=true;});
    viewport.addEventListener('focusin',event=>{const card=event.target.closest('[data-anatomy-task]');if(card&&Number(card.dataset.anatomyTask)!==task.id){pendingFocus=null;snap(tasks.findIndex(t=>t.id===Number(card.dataset.anatomyTask)));}});
    host.addEventListener('keydown',event=>{
      if(!active)return;
      const taskButton=event.target.closest('[data-anatomy-task-go]');
      if(taskButton&&['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){
        event.preventDefault();const index=navButtons.indexOf(taskButton),next=event.key==='Home'?0:event.key==='End'?tasks.length-1:clamp(index+(event.key==='ArrowLeft'?-1:1));
        navButtons[next].focus();navButtons[next].click();return;
      }
      if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();pendingFocus=null;snap(target+(event.key==='ArrowLeft'?-1:1));}
      else if(event.key==='Escape')onInspector(false);
      else if(['+','=','-','0'].includes(event.key)){event.preventDefault();if(event.key==='0')host.querySelector('[data-anatomy-fit]').click();else zoom(event.key==='-'?.8:1.25);}
      else if(['Enter',' '].includes(event.key)&&event.target.closest('[data-anatomy-kind]')){event.preventDefault();event.target.closest('[data-anatomy-kind]').dispatchEvent(new MouseEvent('click',{bubbles:true}));}
    });
    const resizeObserver=new ResizeObserver(()=>{if(active)paint();});resizeObserver.observe(viewport);
    const motionChanged=()=>{paging.reset();if(frame){stop();position=target;velocity=0;paint();commit();}};
    media.addEventListener('change',motionChanged);
    const screenRect=element=>{const r=element.getBoundingClientRect();return {x:r.left,y:r.top,width:r.width,height:r.height};};
    function getTransitionSnapshot(){
      if(!task||!active)return null;const element=cards.find(c=>Number(c.dataset.anatomyTask)===task.id);if(!element)return null;
      return {task:task.id,model:task,frame:screenRect(element),title:screenRect(element.querySelector('h3')),qRows:[...query.querySelectorAll('[data-anatomy-q-row]')].map(screenRect),cells:[...element.querySelectorAll('[data-anatomy-cell]')].map(cell=>({key:cell.dataset.anatomyCell,rect:screenRect(cell)}))};
    }
    return {update,model,getTransitionSnapshot,hideTooltip(){tooltip.hidden=true;},
      destroy(){active=false;stop();resizeObserver.disconnect();media.removeEventListener('change',motionChanged);window.removeEventListener('pointerup',releasePointer);host.replaceChildren();}};
  }
  global.PtoVisualization2Anatomy=Object.freeze({createModel,related,mount,createPagingInput});
})(window);
