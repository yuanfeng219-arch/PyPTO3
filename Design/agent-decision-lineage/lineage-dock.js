/* Compact projection of the existing presentation graph; selection belongs to app.js. */
(()=>{
 const view=window.LINEAGE_VIEW,data=window.LINEAGE_DATA,pane=document.querySelector('main.pto-ide-frame__pane'),full=document.querySelector('.workspace'),scroller=document.querySelector('.graph-host');
 const dock=document.createElement('nav');dock.className='lineage-dock';dock.hidden=true;dock.setAttribute('aria-label','Decision Lineage 导航');full.before(dock);
 const labels=['SPEC','EXPLORE','DESIGN','DEVELOP','OPTIMIZE','OPEN OPPORTUNITIES'];
 const stages=labels.map((label,i)=>({label,items:view.groups.filter(g=>g.stage===i+1).flatMap(g=>g.items.map(([id,title])=>({id,title})))}));
 stages[2].items.push(...view.branches.map(b=>({id:b.id,title:b.title,extra:true})));
 stages[4].items.splice(1,0,{id:view.river.finding.id,title:view.river.finding.label});stages[4].items.push({id:view.river.failed.id,title:view.river.failed.label,failed:true},{id:'d-stop',title:'Stop'});
 stages[5].items=view.river.opportunities.map(n=>({id:n.id,title:n.label,category:n.categoryId}));
 let mode='overview',savedX=0,version=0,picker=null;
 const reduced=()=>matchMedia('(prefers-reduced-motion: reduce)').matches;
 const close=()=>{picker?.remove();picker=null;dock.querySelectorAll('[aria-expanded]').forEach(b=>b.setAttribute('aria-expanded','false'))};
 function render(id){close();dock.replaceChildren();const current=stages.findIndex(s=>s.items.some(n=>n.id===id)),title=data.nodes.find(n=>n.id===id)?.title||id;
 const header=document.createElement('header');header.innerHTML='<strong>Softmax / Decision Lineage</strong><button class="btn btn-ghost" type="button">展开图谱 ↗</button>';header.querySelector('button').onclick=expand;dock.append(header);
 const strip=document.createElement('div');strip.className='dock-strip';dock.append(strip);
 stages.forEach((stage,i)=>{const b=document.createElement('button');b.type='button';b.className='dock-stage'+(i===current?' is-current':'');b.setAttribute('aria-expanded','false');b.title=stage.items.map(n=>n.title).join(' · ');b.innerHTML=`<span>0${i+1} ${stage.label}</span>`;
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 180 34');svg.setAttribute('aria-hidden','true');
 const pts=new Map();const items=stage.items.filter(n=>!n.extra);items.forEach((n,j)=>{let x=15+j*150/Math.max(1,items.length-1),y=17;if(i===4&&n.failed){x=105;y=5}if(i===5){x=n.category==='target-met'?65:135;y=n.id==='x-fusion'?27:9}pts.set(n.id,[x,y])});
 const line=(a,z,cls='')=>{if(!a||!z)return;const path=document.createElementNS(ns,'path');path.setAttribute('d',`M${a[0]} ${a[1]} L${z[0]} ${z[1]}`);path.setAttribute('class',cls);svg.append(path)};
 if(i===4){for(const [a,z] of [['x-base','f-mask'],['f-mask','x-mask'],['x-mask','x-tail'],['x-tail','x-final'],['x-final','d-stop']])line(pts.get(a),pts.get(z));line(pts.get('x-tail'),pts.get('x-if'),'failed')}
 else if(i===5){items.forEach(n=>line([5,17],pts.get(n.id),'untried'))}else items.slice(1).forEach((n,j)=>line(pts.get(items[j].id),pts.get(n.id)));
 items.forEach(n=>{const c=document.createElementNS(ns,'circle');c.setAttribute('cx',pts.get(n.id)[0]);c.setAttribute('cy',pts.get(n.id)[1]);c.setAttribute('r',n.id===id?5:3);c.setAttribute('class',(n.failed?'failed ':i===5?'untried ':'')+(n.id===id?'selected':''));svg.append(c)});b.append(svg);b.onclick=()=>{const was=b.getAttribute('aria-expanded')==='true';close();if(was)return;b.setAttribute('aria-expanded','true');picker=document.createElement('div');picker.className='dock-picker';picker.setAttribute('aria-label',stage.label+' 节点');stage.items.forEach(n=>{const item=document.createElement('button');item.className='btn btn-ghost';item.textContent=(n.failed?'失败 · ':n.category?(view.river.postRunOpportunities.categories.find(c=>c.categoryId===n.category).categoryName+' · 未尝试 · '):'')+n.title;item.setAttribute('aria-pressed',String(n.id===id));item.onclick=()=>window.selectLineageNode(n.id);picker.append(item)});dock.append(picker);picker.style.left=Math.min(b.offsetLeft,Math.max(0,dock.clientWidth-340))+'px';picker.querySelector('button').focus()};strip.append(b)});
 const context=document.createElement('small');context.textContent=`${current>=0?'0'+(current+1)+' '+labels[current]+' → ':''}${title}`;dock.append(context);
 requestAnimationFrame(()=>{const active=strip.querySelector('.is-current');if(active)strip.scrollLeft=Math.max(0,active.offsetLeft-strip.clientWidth/2+active.clientWidth/2)});
 }
 async function inspect(id){const ticket=++version;if(mode==='overview')savedX=scroller.scrollLeft;mode='inspect';render(id);dock.hidden=false;full.style.height=full.getBoundingClientRect().height+'px';full.inert=true;pane.dataset.lineageMode=mode;await new Promise(requestAnimationFrame);full.style.height='0px';await Promise.all(full.getAnimations().map(a=>a.finished.catch(()=>{})));await new Promise(requestAnimationFrame);if(ticket!==version)return;const analysis=document.getElementById('analysis');pane.scrollTo({top:pane.scrollTop+analysis.getBoundingClientRect().top-pane.getBoundingClientRect().top-dock.getBoundingClientRect().height,behavior:reduced()?'auto':'smooth'});document.getElementById('analysis-title').focus({preventScroll:true})}
 function expand(){++version;close();mode='overview';pane.dataset.lineageMode=mode;dock.hidden=true;full.inert=false;full.style.height='auto';scroller.scrollLeft=savedX;pane.scrollTo({top:0,behavior:reduced()?'auto':'smooth'})}
 document.addEventListener('pointerdown',e=>{if(!dock.contains(e.target))close()});document.addEventListener('keydown',e=>{if(e.key==='Escape'&&picker){e.preventDefault();e.stopImmediatePropagation();const active=dock.querySelector('[aria-expanded="true"]');close();active?.focus()}},true);
 window.LineageDock={inspect,expand,get mode(){return mode}};
})();
