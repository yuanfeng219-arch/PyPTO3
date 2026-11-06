// Pass geometry is a structural projection onto the existing Core event rows.
window.INLINE_PASS_TRANSFORM=(()=>{
 let raf=0,position=0,playing=false,autoPending=false;
 const phases=['P19','P23','P25','P28','P30'];
 function stop(){cancelAnimationFrame(raf);playing=false;}
 function mount(g,{y,W,LEFT,first,onBack,zoomFrom}){
  stop();let zoomProgress=zoomFrom?.length?0:1;const zoomMap=new Map((zoomFrom||[]).map(v=>[v.key,v]));const ns='http://www.w3.org/2000/svg';
  const make=(tag,attrs,parent=g)=>{const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);parent.append(n);return n;};
  g.querySelector('.qk-repeat-summary')?.remove();
  // Keep identity at left and round navigation at right; remove duplicate entry and order chips.
  g.querySelectorAll('[data-qk-inline-pass],[data-qk-collapse],[data-qk-nav],.qk-round-control,.qk-round-label,.qk-pair-control,.qk-pair-caption').forEach(n=>n.remove());
  g.querySelectorAll('line').forEach(n=>{if(+n.getAttribute('y1')===y+20&&+n.getAttribute('y2')===y+36)n.remove();});
  g.querySelectorAll('text').forEach(n=>{if(n.textContent==='→'||n.textContent==='执行引擎')n.remove();});
  const chart=make('g',{class:'inline-pass-chart',transform:'translate(0 -24)'},g);
  for(const n of [...g.children]){const yy=Number(n.getAttribute('y')||n.getAttribute('y1')||0);if(n!==chart&&(n.hasAttribute('data-lane-camera')||yy>=y+90))chart.append(n);}
  g.querySelectorAll('text').forEach(n=>{n.style.fontFamily='var(--font)';if(n.closest('[data-event-start]'))n.setAttribute('font-size','12');else if(chart.contains(n))n.setAttribute('font-size','13');});
  make('line',{x1:LEFT-95,x2:W-24,y1:y+63,y2:y+63,stroke:'#383838'},g);

  const controls=make('g',{'class':'inline-pass-controls','aria-label':'核内 Pass 变换',transform:'translate('+((W-440)/2-(LEFT-104))+' -28)'});
  make('rect',{x:LEFT-104,y:y+45,width:440,height:38,rx:19,fill:'#292929'},controls);
  const passLabel=make('text',{x:LEFT+28,y:y+68,fill:'#aaa','font-size':13},controls);passLabel.textContent='pass';
  function button(label,x,width,action){const n=make('g',{role:'button',tabindex:0,'aria-label':label,class:'inline-pass-button'},controls);const r=make('rect',{x,y:y+45,width,height:38,rx:19,fill:'#303030'},n);const t=make('text',{x:x+width/2,y:y+67,'text-anchor':'middle',fill:'#ddd','font-size':12},n);t.textContent=label;n.addEventListener('click',e=>{e.stopPropagation();zoomProgress=1;action();});n.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();e.stopPropagation();zoomProgress=1;action();}});return {n,r,t};}
  const buttons=phases.map((p,i)=>button(p,LEFT+76+i*48,44,()=>{stop();position=i+1;draw();}));
  const play=button('播放',LEFT-94,44,()=>{if(playing){stop();draw();return;}if(position>=5)position=0;playing=true;let last=0;const tick=now=>{if(!g.isConnected){stop();return;}if(last)position=Math.min(5,position+(now-last)/1800);last=now;draw();if(position<5)raf=requestAnimationFrame(tick);else{stop();draw();}};raf=requestAnimationFrame(tick);});
  const replay=button('重播本步',LEFT-44,44,()=>{const target=Math.max(1,Math.ceil(position));stop();position=target-1;let last=0;playing=true;const tick=now=>{if(!g.isConnected){stop();return;}if(last)position=Math.min(target,position+(now-last)/1800);last=now;draw();if(position<target)raf=requestAnimationFrame(tick);else{stop();draw();}};raf=requestAnimationFrame(tick);});
  play.r.setAttribute('fill','transparent');replay.r.setAttribute('fill','transparent');replay.t.textContent='';
  [play,replay].forEach(v=>{v.n.addEventListener('pointerdown',()=>{play.r.setAttribute('fill','transparent');replay.r.setAttribute('fill','transparent');v.r.setAttribute('fill','#383838');});});
  const rcx=LEFT-22,rcy=y+63;
  make('path',{fill:'none',stroke:'#eee','stroke-width':1.8,'stroke-linecap':'round',d:'M'+rcx+' '+(rcy-9)+' A9 9 0 1 1 '+(rcx-7)+' '+(rcy-5)},replay.n);
  make('path',{fill:'#eee',d:'M'+(rcx-5)+' '+(rcy-9)+' l6 -4 v8 z'},replay.n);
  replay.n.querySelectorAll('path').forEach(n=>n.setAttribute('transform','translate('+rcx+' '+rcy+') scale(.75) translate('+(-rcx)+' '+(-rcy)+')'));
  const back=button('返回',W-96-((W-440)/2-(LEFT-104)),64,()=>{stop();onBack?.();});back.n.setAttribute('transform','translate(0 -8)');back.r.setAttribute('fill','#303030');
  const caption=make('title',{},controls);
  const events=[...g.querySelectorAll('[data-event-start]')];
  const originals=new Map(events.map(n=>[n,[...n.querySelectorAll('rect')].map(r=>({r,x:+r.getAttribute('x'),y:+r.getAttribute('y'),w:+r.getAttribute('width')}))]));
  function draw(){
   g.setAttribute("data-zoom-progress",zoomProgress.toFixed(3));
   const lo=Math.floor(position),hi=Math.min(5,lo+1),t=position-lo,a=window.QK_PASS_EVOLUTION.geometry(lo,first),b=window.QK_PASS_EVOLUTION.geometry(hi,first),map=new Map();
   for(const n of events){const op=n.dataset.qkInlineEvent,iter=+n.dataset.fragment,from=a.find(e=>e.op===op&&e.iter===iter),to=b.find(e=>e.op===op&&e.iter===iter);if(!from||!to){n.style.display='none';continue;}n.style.display='';const mix=k=>from[k]+(to[k]-from[k])*t;let x=LEFT+((mix('x')-118)/418)*(W-LEFT-36),w=Math.max(3,mix('w')/418*(W-LEFT-36));const zoomSource=zoomMap.get(op+'-'+iter);if(zoomSource&&zoomProgress<1){x=zoomSource.x+(x-zoomSource.x)*zoomProgress;w=zoomSource.w+(w-zoomSource.w)*zoomProgress;}
    const base=originals.get(n),row=['scale','exp','reduce','send_p'].includes(op)?(position<2?3+(6-3)*Math.max(0,position-1):6):op==='q'?0:op==='mm'||op==='pv'?3:op==='send_score'?4:op==='wait_p'?5:op==='recv_p'?7:op==='write_o'?8:op==='write_ml'?9:null;
    if(row===null)continue;let yy=y+96+row*26;if(zoomSource&&zoomProgress<1)yy=zoomSource.y+(yy-zoomSource.y)*zoomProgress;
    n.style.opacity=String(Math.max(.12,mix('opacity')));base.forEach(({r},i)=>{r.setAttribute('x',x-(i?2:0));r.setAttribute('width',w+(i?4:0));r.setAttribute('y',yy-(i?2:0));});
    let text=n.querySelector('text');if(op==='q'&&position>=1&&!text){text=make('text',{'text-anchor':'middle','font-size':11,fill:'#111','pointer-events':'none'},n);}if(text){text.setAttribute('font-size','12');text.style.fontFamily='var(--font)';if(op==='q')text.textContent=position>=1?(iter===first?'Q 驻留':'复用 Q'):'';text.setAttribute('x',x+w/2);text.setAttribute('y',yy+8.5);text.setAttribute('dominant-baseline','middle');}
    const primary=n.querySelector('rect'),phase=Math.ceil(position);
    const changed=phase===1?op==='q':phase===2?['send_score','scale','exp','reduce','send_p','recv_p','wait_p'].includes(op):phase===3?['scale','exp','reduce'].includes(op):phase===4?(op==='mm'&&iter===first+1||op==='pv'&&iter===first):phase===5?['mm','pv'].includes(op):false;
    primary.setAttribute('fill',changed?'#63e6b3':op==='wait_p'?'#333':iter===first?'#a0a0a0':'#777');primary.setAttribute('stroke','none');primary.setAttribute('stroke-width','0');n.setAttribute('data-pass-changed',String(changed));
    map.set(op+'-'+iter,{x,y:yy+8,w});
   }
   for(const path of g.querySelectorAll('.qk-dependency-link')){const iter=path.dataset.dependencyIter,a=map.get(path.dataset.dependencyFrom+'-'+iter),b=map.get(path.dataset.dependencyTo+'-'+iter);if(a&&b)path.setAttribute('d','M'+(a.x+a.w)+' '+a.y+' H'+((a.x+a.w+b.x)/2)+' V'+b.y+' H'+b.x);path.style.opacity=position<1?.15:1;}
   g.querySelector('.qk-pair-window')?.setAttribute('opacity','0');
   let split=g.querySelector('.inline-split-detail');if(!split)split=make('text',{class:'inline-split-detail',x:LEFT,y:y+408,fill:'#bbb','font-size':12});split.textContent=Math.ceil(position)===3?'AIV subblock 0 有效；subblock 1 · valid_shape = 0':'';
   buttons.forEach((v,i)=>{const on=Math.ceil(position)===i+1;v.r.setAttribute('fill','transparent');v.t.textContent=phases[i].slice(1);v.t.setAttribute('fill','#fff');v.t.setAttribute('font-size','14');v.n.setAttribute('aria-pressed',on);let line=v.n.querySelector('.pass-underline');if(!line)line=make('line',{class:'pass-underline',x1:LEFT+86+i*48,x2:LEFT+108+i*48,y1:y+73,y2:y+73,stroke:'#eee','stroke-width':1.5},v.n);line.setAttribute('opacity',on?1:.7);});
   play.t.textContent='';play.n.setAttribute('aria-label',playing?'暂停':'播放');
   let icon=play.n.querySelector('.transport-icon');if(!icon)icon=make('path',{class:'transport-icon',fill:'#eee',stroke:'#eee','stroke-width':1.8,'stroke-linejoin':'round'},play.n);
   icon.setAttribute('fill',playing?'none':'#eee');const cx=LEFT-72,cy=y+63;icon.setAttribute('transform','translate('+cx+' '+cy+') scale(.8) translate('+(-cx)+' '+(-cy)+')');icon.setAttribute('d',playing?'M'+(cx-4)+' '+(cy-6)+'v12M'+(cx+4)+' '+(cy-6)+'v12':'M'+(cx-4)+' '+(cy-6)+'l10 6-10 6z');
   caption.textContent=['变换前','Q 外提 / 复用','AIC · AIV 拆分','子块有效形状','QK₁ 前移 · PV₀ 等 P₀','L0 内部展开'][Math.ceil(position)];
   let detail=g.querySelector('.inline-pass-detail');if(!detail)detail=make('text',{class:'inline-pass-detail',x:LEFT,y:y+382,fill:'#bbb','font-size':12});detail.textContent='';
   g.querySelector('.inline-l0-detail')?.remove();
  }
  const begin=()=>{if(autoPending){autoPending=false;position=0;play.n.dispatchEvent(new MouseEvent('click',{bubbles:false}));}};
  draw();if(zoomProgress<1){let start;const zoom=now=>{if(!g.isConnected)return;if(!start)start=now;const t=Math.min(1,(now-start)/700);zoomProgress=1-Math.pow(1-t,3);draw();if(t<1)raf=requestAnimationFrame(zoom);else begin();};raf=requestAnimationFrame(zoom);}else begin();return {stop};
 }
 return {mount,stop,autoplay:()=>{autoPending=true;}};
})();
