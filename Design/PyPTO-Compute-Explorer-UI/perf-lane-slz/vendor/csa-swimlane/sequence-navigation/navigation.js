(function(global){
  function mount(root,{items,variant='stage',full=[0,1],onSelect=()=>{},onFocus=()=>{},onRange=()=>{}}){
    const abort=new AbortController(),on=(n,k,f)=>n.addEventListener(k,f,{signal:abort.signal});
    root.classList.add('sequence-nav');root.dataset.variant=variant;
    let current=0,range=[...full],drag=null;
    const width=full[1]-full[0],pct=t=>(t-full[0])/width*100;
    const buttons=items.map((item,i)=>{const b=document.createElement('button');b.type='button';b.setAttribute('aria-label',item.label+(item.detail?' · '+item.detail:''));b.title=item.label+(item.detail?'\n'+item.detail:'');if(item.kind==='prefill'||item.label==='Prefill')b.dataset.kind='prefill';if(item.alert)b.dataset.alert='true';
      if(variant==='time'){b.style.left=`calc(${pct(item.start)}% + 1.5px)`;b.style.width=`max(1px, calc(${(item.end-item.start)/width*100}% - 3px))`;}
      const label=document.createElement('span');label.className='sequence-label';label.textContent=item.label;b.append(label);root.append(b);on(b,'click',()=>onSelect(i));on(b,'dblclick',()=>onFocus(i));return b;});
    const window=document.createElement('div');window.className='sequence-window';window.tabIndex=0;window.setAttribute('role','group');window.setAttribute('aria-label','Visible time range. Arrow keys to pan; plus or minus to zoom.');
    for(const edge of ['start','end']){const h=document.createElement('span');h.className='sequence-handle';h.dataset.edge=edge;window.append(h);}
    if(variant==='time')root.append(window);
    function clamp(a,b){const w=Math.min(width,Math.max(width/1500,b-a)),lo=Math.max(full[0],Math.min(full[1]-w,a));return [lo,lo+w];}
    function update({index=current,range:next=range}={}){current=index;range=[...next];buttons.forEach((b,i)=>{if(i===index)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});window.style.left=pct(range[0])+'%';window.style.width=(range[1]-range[0])/width*100+'%';window.setAttribute('aria-description',range.map(x=>(x/1000).toFixed(2)+' s').join(' to '));}
    on(window,'pointerdown',e=>{e.preventDefault();e.stopPropagation();drag={x:e.clientX,range:[...range],edge:e.target.dataset.edge};window.setPointerCapture(e.pointerId);});
    on(window,'pointermove',e=>{if(!drag)return;const d=(e.clientX-drag.x)/root.clientWidth*width;let [a,b]=drag.range;if(drag.edge==='start')a=Math.max(full[0],Math.min(b-width/1500,a+d));else if(drag.edge==='end')b=Math.min(full[1],Math.max(a+width/1500,b+d));else{a+=d;b+=d;}onRange(clamp(a,b));});
    const itemAt=e=>{const t=full[0]+(e.clientX-root.getBoundingClientRect().left)/root.clientWidth*width;return items.findIndex(i=>t>=i.start&&t<i.end);};
    on(window,'pointerup',e=>{if(drag&&!drag.edge&&Math.abs(e.clientX-drag.x)<4){const i=itemAt(e);if(i>=0)onSelect(i);}drag=null;});
    on(window,'pointercancel',()=>drag=null);
    on(window,'dblclick',e=>{if(e.target.dataset.edge)return;const i=itemAt(e);if(i>=0)onFocus(i);});
    on(root,'keydown',e=>{if(e.target===window){let [a,b]=range;const w=b-a;if(e.key==='ArrowLeft'){a-=w*.15;b-=w*.15;}else if(e.key==='ArrowRight'){a+=w*.15;b+=w*.15;}else if(['+','=','-'].includes(e.key)){const f=e.key==='-'?1.25:.8;a+=(w-w*f)/2;b-=(w-w*f)/2;}else return;e.preventDefault();e.stopPropagation();onRange(clamp(a,b));}else if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();e.stopPropagation();const i=Math.max(0,Math.min(items.length-1,current+(e.key==='ArrowRight'?1:-1)));onSelect(i);buttons[i].focus();}});
    update();return {update,destroy(){abort.abort();root.replaceChildren();root.classList.remove('sequence-nav');}};
  }
  global.SequenceNavigation={mount};
})(globalThis);
