/* Shared-axis critical path adapter. No separate zoom or lane renderer. */
window.CriticalPathIntegration = (() => {
  let api, nav, drawer, rank, expanded=new Set(), opened=false, height=120, savedRange=null;
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>'&#'+c.charCodeAt(0)+';');
  const ms=n=>(Number(n||0)/1000).toFixed(3);
  const wait=p=>/(^|_)wait($|_)/.test(p.name||p.leaf||'');
  const task=(r,id)=>r.criticalPath.find(p=>String(p.taskId)===String(id))||r.launches.find(p=>String(p.taskId)===String(id));
  function choose(id){opened=true;api.select(String(id));refresh();api.redraw();}
  function resize(h){height=Math.max(120,Math.min(document.querySelector('.main-column').clientHeight*.5,h));drawer.style.height='auto';document.body.style.setProperty('--cpi-drawer-height',height+'px');api.redraw();}
  function mount(adapter){
    api=adapter;document.body.classList.add('cpi-host');
    nav=document.createElement('div');nav.className='cpi-nav';nav.setAttribute('aria-label','关键路径时间导航');document.querySelector('.lane-axis-sticky').prepend(nav);
    drawer=document.createElement('section');drawer.className='cpi-drawer';drawer.hidden=true;drawer.setAttribute('aria-label','局部依赖');document.querySelector('#path-analysis-lane').append(drawer);
    nav.addEventListener('pointerdown',e=>{if(e.target.closest('button'))e.stopPropagation()});
    nav.addEventListener('dblclick',e=>{if(e.target.closest('button'))e.stopPropagation()});
    nav.onclick=e=>{const b=e.target.closest('button');if(!b)return;
      if(b.dataset.task)choose(b.dataset.task);
      if(b.dataset.gap){const [a,z]=b.dataset.gap.split(':').map(Number),s=api.state(),p=s.rank.criticalPath;if(!expanded.size)savedRange=s.range?.slice()||[0,s.rank.kpi.windowUs];for(let i=a;i<=z;i++)expanded.add(i);
        const lo=p[a].startUs,hi=p[z].endUs,full=s.rank.kpi.windowUs,current=s.range||[0,full];
        const span=Math.max(full/256,(current[1]-current[0])/4,Math.min(current[1]-current[0],(hi-lo)*1.3));api.range((lo+hi-span)/2,(lo+hi+span)/2);
      }
      if(b.dataset.collapse){expanded.clear();const range=savedRange||[0,api.state().rank.kpi.windowUs];savedRange=null;api.range(...range)}
      if(b.dataset.info){const h=nav.querySelector('.cpi-help');h.hidden=!h.hidden;b.setAttribute('aria-expanded',String(!h.hidden))}
    };
    nav.addEventListener('pointerover',e=>{const b=e.target.closest('[data-task]');if(b)api.hover(b.dataset.task)});nav.addEventListener('pointerleave',()=>api.hover(null));
    drawer.onclick=e=>{const b=e.target.closest('button');if(b?.dataset.task)choose(b.dataset.task);if(b?.dataset.close){opened=false;drawer.hidden=true;api.redraw()}};
    drawer.addEventListener('pointerover',e=>{const b=e.target.closest('[data-task]');if(b)api.hover(b.dataset.task);});
    drawer.addEventListener('pointerleave',()=>api.hover(null));
    drawer.addEventListener('focusin',e=>{const b=e.target.closest('[data-task]');if(b)api.hover(b.dataset.task);});
    drawer.addEventListener('focusout',()=>api.hover(null));
    let drag=null;
    drawer.addEventListener('pointerdown',e=>{if(!e.target.closest('.cpi-resize'))return;e.preventDefault();drag={y:e.clientY,h:height};e.target.setPointerCapture(e.pointerId)});
    drawer.addEventListener('pointermove',e=>{if(drag)resize(drag.h+drag.y-e.clientY)});drawer.addEventListener('pointerup',()=>drag=null);drawer.addEventListener('pointercancel',()=>drag=null);
    drawer.addEventListener('keydown',e=>{if(!e.target.matches('.cpi-resize'))return;if(['ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();resize(height+(e.key==='ArrowUp'?20:-20))}});
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&opened){opened=false;drawer.hidden=true;api.redraw()}});
  }
  function refresh(){
    if(!api)return;const s=api.state(),r=s.rank;if(!r)return;
    if(rank!==r){rank=r;expanded.clear();opened=false;savedRange=null}
    const p=r.criticalPath,selected=p.findIndex(x=>x.ids?.includes(s.selected)||String(x.taskId)===String(r.launches.find(l=>l.id===s.selected)?.taskId));
    const range=s.range||[0,r.kpi.windowUs],width=nav.clientWidth-s.left-12,x=i=>(p[i].startUs-range[0])/(range[1]-range[0])*width;
    const anchors=[0,p.length-1,...p.map((v,i)=>({v,i})).sort((a,b)=>b.v.spanUs-a.v.spanUs).slice(0,3).map(v=>v.i),...expanded];if(selected>=0)anchors.push(selected);
    const ids=[...new Set(anchors)].sort((a,b)=>a-b).filter(i=>x(i)>=0&&x(i)<=width),shown=[];
    for(const i of ids){if(!shown.length||x(i)-x(shown.at(-1))>=52)shown.push(i);else if(i===selected)shown[shown.length-1]=i}
    let html='';for(let k=0;k<shown.length;k++){const i=shown[k],v=p[i],px=Math.max(24,Math.min(width-24,x(i)));html+=`<button class="cpi-tag ${wait(v)?'wait':''} ${i===selected?'selected':''}" data-task="${esc(v.taskId)}" style="left:${px}px" data-tooltip="${esc(v.name)} · ${ms(v.spanUs)} ms\n开始 ${ms(v.startUs)} ms">#${String(i+1).padStart(2,'0')}</button>`;
      const next=shown[k+1];if(next!==undefined&&next-i>1){html+=`<button class="cpi-gap" data-gap="${i+1}:${next-1}" style="left:${(x(i)+x(next))/2}px;${x(next)-x(i)<140?'top:0;font-size:10px':''}">+${next-i-1} 项</button>`}}
    nav.innerHTML=`<span class="cpi-label"><button class="cpi-info" data-info="1" aria-label="关键路径说明" aria-expanded="false"><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" aria-hidden="true"><path d="M3 3v5a3 3 0 0 0 3 3h5M3 3h8"/><circle cx="3" cy="3" r="1.5" fill="#262626"/><circle cx="11" cy="3" r="1.5" fill="#262626"/><circle cx="11" cy="11" r="1.5" fill="#262626"/></svg></button>关键路径</span><div class="cpi-track" style="left:${s.left}px;right:12px">${html}</div>${expanded.size?'<button class="cpi-collapse" data-collapse="1">收起展开</button>':''}<div class="cpi-help" hidden>关键路径表示沿任务依赖串起、覆盖执行时间最长的一条候选链。当前数据沿依赖边比较各条链从最早开始到最晚结束的时间跨度，保留跨度最大的前驱，再回溯提取路径。它不是任务耗时的简单相加：重叠时间不重复计算，间隙会计入。可用它追踪影响完成时间的依赖；并不意味着链上每个任务都已被证明是性能瓶颈。</div>`;
    drawer.hidden=!opened;if(!opened)return;
    const id=r.launches.find(l=>l.id===s.selected)?.taskId,v=task(r,id);if(!v){drawer.hidden=true;return}
    const edges=r.edges||[],neighbors=(pred)=>[...new Set(edges.filter(e=>String(pred?e.succ:e.pred)===String(id)).map(e=>String(pred?e.pred:e.succ)))];
    const side=pred=>neighbors(pred).map(n=>{const t=task(r,n);return `<button data-task="${esc(n)}" data-tooltip="${esc(t?.name||t?.leaf||'Task '+n)}" ${t?'':'disabled'}>${esc(t?.name||t?.leaf||'Task '+n)}</button>`}).join('')||'<small>无直接依赖</small>';
    // Keep the resize handle mounted while pointer capture is active.
    if(!drawer.firstChild)drawer.innerHTML='<header></header><div class="cpi-detail"></div>';
    drawer.style.height='auto';document.body.style.setProperty('--cpi-drawer-height',height+'px');drawer.querySelector('header').innerHTML=`<button data-close="1" aria-label="收起关键路径" data-tooltip="收起关键路径">×</button>`;
    drawer.querySelector('.cpi-detail').innerHTML=`<div class="cpi-neighbors">${side(true)}</div><div class="cpi-center" data-task="${esc(id)}"><strong>${esc(v.name||v.leaf)}</strong><b>${ms(v.spanUs??v.endUs-v.startUs)} <small>ms</small></b><p>实测跨度</p></div><div class="cpi-neighbors">${side(false)}</div>`;
  }
  function hover(id){
    if(!nav)return;
    nav.querySelectorAll('.cpi-hover').forEach(n=>n.classList.remove('cpi-hover'));
    nav.querySelectorAll('[data-hover-preview]').forEach(n=>n.remove());
    if(id==null)return;
    const existing=[...nav.querySelectorAll('[data-task]')].find(n=>n.dataset.task===String(id));
    if(existing){existing.classList.add('cpi-hover');return;}
    const s=api.state(),i=s.rank.criticalPath.findIndex(p=>String(p.taskId)===String(id));if(i<0)return;
    const p=s.rank.criticalPath[i],range=s.range||[0,s.rank.kpi.windowUs],track=nav.querySelector('.cpi-track'),width=track.clientWidth;
    const tag=document.createElement('button');tag.className='cpi-tag cpi-hover';tag.dataset.hoverPreview='true';tag.dataset.task=String(id);tag.textContent='#'+String(i+1).padStart(2,'0');tag.style.left=Math.max(24,Math.min(width-24,(p.startUs-range[0])/(range[1]-range[0])*width))+'px';tag.dataset.tooltip=p.name;track.append(tag);
  }
  return {mount,refresh,choose,hover};
})();
