(() => {
 const pages=['index.html','intro.html','operator.html'];
 const current=Math.max(0,pages.indexOf(location.pathname.split('/').pop()||'index.html'));
 function go(delta){const next=current+delta;if(next>=0&&next<pages.length)location.href=pages[next];}
 document.addEventListener('keydown',event=>{
  if(event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
  if(event.target.closest('input,textarea,select,[contenteditable="true"],dialog,[role="dialog"]'))return;
  if(['ArrowRight','ArrowDown','ArrowLeft','ArrowUp'].includes(event.key)){
   event.preventDefault();event.stopPropagation();go(event.key==='ArrowRight'||event.key==='ArrowDown'?1:-1);
  }
 },true);
 const nav=document.createElement('nav');nav.className='tour-pages';nav.setAttribute('aria-label','首页翻页');
 nav.innerHTML=pages.map((page,i)=>`<a href="${page}" ${i===current?'aria-current="page"':''} aria-label="${['计算领域体验设计创新','项目背景','昇腾算子开发'][i]}">${i+1}</a>`).join('');document.body.append(nav);
})();