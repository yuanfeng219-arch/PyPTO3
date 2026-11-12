/* Fit the shared compile viewer through its existing action; local to Lab Tour. */
(()=>{
 const drawer=window.CompileDrawer;if(!drawer?.open)return;
 const open=drawer.open;let cancelPending;
 drawer.open=function(...args){
  cancelPending?.();const result=open.apply(this,args);
  const host=document.querySelector('.compile-linked-host'),root=host?.shadowRoot;
  if(!root)return result;
  if(!root.querySelector('[data-lab-fit-style]')){
   const style=document.createElement('style');style.dataset.labFitStyle='';
   style.textContent=':host(.lab-fit-pending) .graph{opacity:0}';root.append(style);
  }
  host.classList.add('lab-fit-pending');let timer=0,done=false,readyAt=0;const started=performance.now();
  const finish=()=>{
   if(done)return;done=true;clearTimeout(timer);host.classList.remove('lab-fit-pending');
   root.removeEventListener('pointerdown',finish);root.removeEventListener('wheel',finish);
   if(cancelPending===finish)cancelPending=undefined;
  };
  cancelPending=finish;
  root.addEventListener('pointerdown',finish,{once:true});root.addEventListener('wheel',finish,{once:true,passive:true});
  const fit=()=>{
   if(done)return;if(host.hidden||!host.isConnected){finish();return;}
   const graph=root.querySelector('.graph');
   const button=[...root.querySelectorAll('.linked-panel > header button')].find(node=>node.textContent.trim()==='适配');
   const nodes=[...root.querySelectorAll('.react-flow__node')];
   const ready=graph?.clientWidth&&graph.clientHeight&&nodes.length&&nodes.every(node=>node.offsetWidth&&node.offsetHeight&&node.style.visibility!=='hidden')&&button;
   if(ready){
    if(!readyAt)readyAt=performance.now();
    if(performance.now()-readyAt>=160){button.click();requestAnimationFrame(()=>requestAnimationFrame(finish));return;}
   }else readyAt=0;
   if(performance.now()-started<3000){timer=setTimeout(fit,80);}else finish();
  };
  // Run after the viewer's initial 120ms selection zoom, using the final split dimensions.
  timer=setTimeout(fit,260);return result;
 };
})();

/* 默认结论：进页面先说清「最值得看的是哪一处」，不让一屏数据自己说话。
 * 结论不另算：页面已经按「关键路径关联优先 + 分布异常补位」排过序（工具栏「优化」里的优化候选），
 * 这里取它的第 1 条，再从关键路径数据找同名的那一步，补上它在关键路径上的位置和跨度，
 * 全部取页面已经渲染出来的数字，不会和下面的泳道、菜单对不上。
 * 切 rank 或重画工具栏后跟着重建；只在本目录生效，../device-perf-lane/ 不动。 */
(()=>{
 const CHOICE='.bar-menu[data-menu="results"] [data-result-choice]';
 const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const text=(root,sel)=>root?.querySelector(sel)?.textContent.replace(/\s+/g,' ').trim()||'';

 // 这一步给关键路径新增了多少时间：取下面关键路径条自己画出来的那段说明，
 // 除以顶栏的执行窗口得到占比。路径模式关掉时条会消失，所以按 rank + 步骤缓存。
 const shareCache=new Map();
 function pathShare(id){
  const rank=document.getElementById('rank')?.value||'',key=rank+'|'+id;
  if(shareCache.has(key))return shareCache.get(key);
  const seg=document.querySelector(`#cp .cp-seg[data-t="${CSS.escape(id)}"]`);
  const note=seg?.querySelector('desc')?.textContent||seg?.dataset.tooltip||'';
  const added=note.match(/路径新增时间\s*([\d.]+)\s*ms/);
  const window=(document.getElementById('stats-toggle')?.textContent||'').match(/([\d.]+)\s*ms/);
  if(!added||!window)return null;
  const pct=Number(added[1])/Number(window[1])*100;
  const share={ms:added[1]+' ms',pct:Math.max(1,Math.min(100,Math.round(pct)))};
  shareCache.set(key,share);return share;
 }

 // 关键路径里的同名步骤：序号、总步数、跨度，以及它是不是路径上跨度最长的一步。
 function pathInfo(kernel){
  const rows=window.ResultPathRows?.() || [];
  if(!rows.length)return null;
  const hit=rows.find(row=>row.name===kernel)||rows.find(row=>row.name.includes(kernel));
  if(!hit)return null;
  return {hit,total:rows.length,longest:rows.every(row=>row.us<=hit.us),share:pathShare(hit.id)};
 }

 function model(){
  const choices=[...document.querySelectorAll(CHOICE)],top=choices[0];
  if(!top)return null;
  const kernel=text(top,'.result-name');if(!kernel)return null;
  const icon=top.querySelector('.semantic-icon');
  return {key:top.dataset.resultChoice,kernel,
          reason:text(top,'small'),rest:choices.length-1,
          color:icon?.style.background||'#bca0f3',
          icon:icon?.innerHTML||'',path:pathInfo(kernel)};
 }

 // 一项「标题 + 数值 + 图形」，并排排在右侧；图形只作视觉，读屏走标题与数值。
 function metric(label,value,figure,note){
  return `<div class="lead-metric"><span>${esc(label)}</span><b>${esc(value)}</b>`
   +(figure?`<div class="lead-fig" aria-hidden="true">${figure}</div>`:'')
   +(note?`<em>${esc(note)}</em>`:'')+`</div>`;
 }

 // 特征本身也画出来：块差距画成两条对比条，占比类画成一条填充条。
 function featureMetric(reason){
  const ratio=reason.match(/([\d.]+)\s*×/),percent=reason.match(/(\d+(?:\.\d+)?)\s*%/);
  if(ratio){
   const times=Math.max(1,Number(ratio[1]));
   return metric('最慢块 / 平均块',ratio[0],
    `<div class="lead-pair"><i style="width:${(100/times).toFixed(1)}%"></i>`
    +`<i class="is-lead" style="width:100%"></i></div>`);
  }
  if(percent){
   return metric(reason.replace(percent[0],'').replace(/[:：·]\s*$/,'').trim()||'占比',percent[0],
    `<div class="lead-fill"><i class="is-lead" style="width:${Math.min(100,Number(percent[1]))}%"></i></div>`);
  }
  return metric('已标记特征',reason,'');
 }

 function markup(m){
  const path=m.path,name=path?path.hit.name:m.kernel;
  return `<div class="lead-id">`
   +(m.icon?`<span class="lead-icon" aria-hidden="true">${m.icon}</span>`:'')
   +`<b>${esc(name)}</b>`
   +`<span class="lead-place">${esc(path?`关键路径 第 ${path.hit.index} / ${path.total} 步`:'路径外 · 分布异常最明显')}</span>`
   +(path?.hit.feature?`<em class="lead-tag">${esc(path.hit.feature)}</em>`:'')+`</div>`
   +`<div class="lead-metrics">`
   +(path?metric('跨度',path.hit.ms,'',path.longest?'路径上最长':''):'')
   +(path?.share?metric('占关键路径',path.share.pct+'%',
      `<div class="lead-fill"><i class="is-lead" style="width:${path.share.pct}%"></i></div>`,
      '新增 '+path.share.ms):'')
   +(m.reason?featureMetric(m.reason):'')
   +`</div>`;
 }

 // 结论指到的那一步，在下面真实的关键路径条上点亮，省掉一条重复的迷你路径。
 function markPath(id){
  const bar=document.getElementById('cp');if(!bar)return;
  bar.querySelectorAll('.cp-lead').forEach(node=>node.classList.remove('cp-lead'));
  if(id)bar.querySelector(`.cp-seg[data-t="${CSS.escape(id)}"]`)?.classList.add('cp-lead');
 }

 function mount(){
  const column=document.querySelector('.main-column'),header=document.getElementById('run-header');
  if(!column||!header)return;
  let host=document.getElementById('lead-finding'),last='';
  const render=()=>{
   const m=model();
   if(!m){if(host)host.hidden=true;return;}
   const html=markup(m);
   if(!host){
    host=document.createElement('section');host.id='lead-finding';host.className='lead-finding';
    host.setAttribute('aria-label','默认结论：最值得先看的一处');
    host.setAttribute('role','button');host.tabIndex=0;
    const activate=()=>{window.focusLeadFinding?.(host.dataset.leadKey,host.dataset.pathId);render();};
    host.addEventListener('click',activate);
    host.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();activate();}});
    header.after(host);
   }
   host.hidden=false;host.dataset.leadKey=m.key;host.dataset.pathId=m.path?.hit.id||'';
   host.setAttribute('aria-pressed',String(!!window.isLeadFindingFocused?.()));
   document.documentElement.style.setProperty('--lead-color',m.color);
   if(html!==last){host.innerHTML=html;last=html;}
   markPath(m.path?.hit.id);
  };
  render();
  // 工具栏每次重画（切 rank、换特征、缩放）后菜单会整块替换，结论跟着重取；
  // 关键路径条重画后点亮也要补回去（只改 class，不会再触发这里的 childList）。
  let queued=0;
  const again=()=>{if(queued)return;queued=setTimeout(()=>{queued=0;render();},32);};
  const watch=node=>node&&new MutationObserver(again).observe(node,{childList:true,subtree:true});
  watch(document.getElementById('lane-ovl'));
  watch(document.getElementById('cp'));
 }
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>setTimeout(mount,0));
 else setTimeout(mount,0);
})();

/* 首屏评审里的三处 P1：泳道底图自己不说话、八种语义色没有常驻图例、「分析」看着像禁用按钮。
 * 前两处都只在本目录叠一层：block 的明度在渲染后改，图例复用页面自己那份内容，不动 ../device-perf-lane/。 */
(()=>{
 const BASE='#565656';                        // 页面给 Task block 的底色
 const GRADES=['#4e4e4e','#5e5e5e','#747474'];// 短 / 中 / 长，三档围着原色走，不另起一套灰

 // 四千多个 block 原本同一个灰，哪里慢全靠浮标签指。按当前视图里的块宽（= 块耗时）
 // 分三档明度：多数维持原色，最长的一成提亮，热区自己浮出来。
 function shadeBlocks(){
  const lane=document.getElementById('lane');if(!lane)return;
  const rects=lane.querySelectorAll(`rect[fill="${BASE}"],rect[data-lead-shade]`);
  if(rects.length<200)return;                 // 只在整屏 block 场里做，单任务视图不碰
  const widths=[];rects.forEach(rect=>widths.push(Number(rect.getAttribute('width'))||0));
  const sorted=widths.slice().sort((a,b)=>a-b);
  const at=p=>sorted[Math.floor((sorted.length-1)*p)];
  const mid=at(.75),top=at(.95);
  if(!(top>mid))return;                       // 块宽没有区分度时保持原样
  rects.forEach((rect,i)=>{
   rect.setAttribute('fill',GRADES[widths[i]>top?2:(widths[i]>mid?1:0)]);
   rect.dataset.leadShade='';
  });
 }

 // 常驻图例：取页面自己那份图例内容（原本藏在关闭的菜单里），压成页底一行。
 // 色块原样保留，只把几条长说明缩短、把读图方法留给菜单里的完整版，好让一行放得下。
 // 菜单那份仍在，所以这条对读屏隐藏，避免重复朗读。
 const SHORTEN=[
  [/^Task block/,'Task block · 明暗 = 块耗时相对长短',true],
  [/^候选任务组/,'候选任务组'],
  [/^并行宽度低于/,'并行宽度 25% 参考线'],
 ];
 function legend(){
  const source=document.getElementById('lane-lgnd'),column=document.querySelector('.main-column');
  if(!source||!column||!source.children.length)return;
  let strip=document.getElementById('lead-legend');
  if(!strip){
   strip=document.createElement('div');strip.id='lead-legend';strip.className='lgnd lead-legend';
   strip.setAttribute('aria-hidden','true');column.append(strip);
  }
  const draft=document.createElement('div');
  [...source.children].forEach(node=>{
   const label=node.textContent.replace(/\s+/g,' ').trim();
   if(/^每行 =/.test(label))return;            // 读图方法只留在菜单里的完整版
   const copy=node.cloneNode(true),edit=SHORTEN.find(([test])=>test.test(label));
   if(edit){
    const swatch=copy.querySelector('i');
    copy.textContent='';
    if(swatch){
     if(edit[2]){swatch.removeAttribute('style');swatch.className='lead-ramp';}
     copy.append(swatch);
    }
    copy.append(document.createTextNode(edit[1]));
   }
   draft.append(copy);
  });
  if(strip.innerHTML!==draft.innerHTML)strip.innerHTML=draft.innerHTML;
 }

 function run(){shadeBlocks();legend();}
 let queued=0;
 const again=()=>{if(queued)return;queued=setTimeout(()=>{queued=0;run();},48);};
 function mount(){
  run();
  const watch=node=>node&&new MutationObserver(again).observe(node,{childList:true,subtree:true});
  watch(document.getElementById('lane'));      // 泳道重画后重新分级
  watch(document.getElementById('lane-ovl'));  // 图例内容跟着工具栏重建
 }
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>setTimeout(mount,0));
 else setTimeout(mount,0);
})();
