(() => {
 const root=document.querySelector('.cols');if(!root)return;
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.classList.add('intro-relations');svg.setAttribute('aria-hidden','true');root.append(svg);
 function draw(){
  const r=root.getBoundingClientRect(),plate=root.querySelector('article.layer-tile').getBoundingClientRect();
  const people=[...root.querySelectorAll('.ava')].map(el=>el.getBoundingClientRect());
  if(matchMedia('(max-width:1080px)').matches){svg.innerHTML='';return;}
  const x=plate.right-r.left-10,y=(people[0].top+people[0].bottom+people[1].top+people[1].bottom)/4-r.top;
  const end=people.map(a=>({x:a.left-r.left-9,y:(a.top+a.bottom)/2-r.top}));
  const branch=Math.max(x+22,Math.min(...end.map(a=>a.x))-34);
  const junctionRadius=Math.max(0,Math.min(12,branch-x,...end.map(a=>Math.abs(a.y-y)/2)));
  function roundedBranch(a){
   const radius=Math.min(12,Math.abs(a.y-y)-junctionRadius,Math.max(0,a.x-branch));
   const direction=Math.sign(a.y-y);
   return `M${branch-junctionRadius} ${y}Q${branch} ${y} ${branch} ${y+direction*junctionRadius}V${a.y-direction*radius}Q${branch} ${a.y} ${branch+radius} ${a.y}H${a.x}`;
  }
  svg.setAttribute('viewBox',`0 0 ${r.width} ${r.height}`);
  svg.innerHTML=`<defs><marker id="relation-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1L8 5L1 9" fill="none" stroke="#c7d7db" stroke-width="1.5"/></marker></defs><path d="M${x} ${y}H${branch-junctionRadius}" fill="none" stroke="#c7d7db" stroke-opacity=".6" stroke-width="1.2"/>`+end.map(a=>`<path d="${roundedBranch(a)}" fill="none" stroke="#c7d7db" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#relation-arrow)"/>`).join('')+`<text x="${branch-12}" y="${y-14}" text-anchor="end" fill="#cbd4df" font-family="system-ui" font-size="13" paint-order="stroke" stroke="#090b10" stroke-width="5">两个断点</text>`;
 }
 let frame;const update=()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(draw)};
 window.addEventListener('resize',update);window.addEventListener('load',update);update();setTimeout(update,1200);
})();
