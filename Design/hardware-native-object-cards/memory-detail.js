(() => {
  const dialog = document.getElementById('memory-dialog');
  const close = document.getElementById('memory-close');
  const play = document.getElementById('memory-play');
  const previous = document.getElementById('memory-prev');
  const next = document.getElementById('memory-next');
  const range = document.getElementById('memory-range');
  const steps = document.getElementById('memory-steps');
  const current = document.getElementById('memory-current');
  const over = document.getElementById('memory-over');
  const count = document.getElementById('memory-step-count');
  const status = document.getElementById('memory-status');
  const segments = ['memory-user','memory-auto','memory-system'].map(id=>document.getElementById(id));
  const amounts = [147456,12544,32768];
  const labels = ['起点','用户','自动','系统'];
  const messages = ['从空预算开始。','累计用户声明的占用。','叠加自动生成的占用。','叠加系统占用，超过可用上限 4,352 B。'];
  const spaces = [
    {id:'vec',name:'Vec',hardware:'UB',capacity:'188,416 B 可用',purpose:'Vector 计算与普通 Tile 加载'},
    {id:'mat',name:'Mat',hardware:'L1 / CBUF',capacity:'512 KiB',purpose:'Cube 操作数暂存与矩阵中间结果'},
    {id:'left',name:'Left',hardware:'L0A',capacity:'64 KiB',purpose:'Matmul 左操作数'},
    {id:'right',name:'Right',hardware:'L0B',capacity:'64 KiB',purpose:'Matmul 右操作数'},
    {id:'acc',name:'Acc',hardware:'L0C',capacity:'128 KiB',purpose:'Matmul 累加器'},
    {id:'bias',name:'Bias',hardware:'Bias 缓冲',capacity:'待确认',purpose:'Matmul bias'},
    {id:'scalar',name:'ScalarLocal',hardware:'标量寄存器 / C 栈',capacity:'待确认',purpose:'标量与 ArrayType'}
  ];
  const spaceGrid = document.getElementById('memory-spaces');
  const ubDetail = document.getElementById('memory-ub-detail');
  const otherDetail = document.getElementById('memory-other-detail');
  spaceGrid.innerHTML=spaces.map(space=>`<button class="memory-space" type="button" data-space="${space.id}" aria-pressed="${space.id==='vec'}"><b>${space.name}</b><span>${space.hardware}</span><small>${space.capacity}</small></button>`).join('');
  let step = 0, timer = null;
  steps.innerHTML = labels.map((label,index)=>`<button type="button" data-step="${index}">${label}</button>`).join('');
  function stop(){ if(timer!==null) clearInterval(timer); timer=null; play.textContent=step===3?'重播':'播放'; }
  function render(){
    const total = amounts.slice(0,step).reduce((sum,n)=>sum+n,0);
    segments.forEach((el,i)=>{el.style.width=step>i?`${amounts[i]/210000*100}%`:'0%';});
    current.textContent=`${total.toLocaleString('en-US')} B`;
    const excess=Math.max(0,total-188416);
    over.textContent=excess?`+${excess.toLocaleString('en-US')} B`:'—';
    over.classList.toggle('is-over',excess>0);
    range.value=String(step); count.textContent=`${step} / 3`; status.textContent=messages[step];
    previous.disabled=step===0; next.disabled=step===3;
    steps.querySelectorAll('button').forEach((button,i)=>{button.classList.toggle('active',i===step);button.classList.toggle('passed',i<step);button.setAttribute('aria-current',i===step?'step':'false');});
    if(timer===null) play.textContent=step===3?'重播':'播放';
  }
  function setStep(value){step=Math.max(0,Math.min(3,Number(value)));render();}
  function selectSpace(id){
    const selected=spaces.find(space=>space.id===id);
    if(!selected)return;
    stop();
    ubDetail.hidden=id!=='vec';otherDetail.hidden=id==='vec';
    spaceGrid.querySelectorAll('button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.space===id)));
    if(id!=='vec'){
      document.getElementById('memory-other-title').textContent=selected.name+' / '+selected.hardware;
      document.getElementById('memory-other-purpose').textContent=selected.purpose;
      document.getElementById('memory-other-capacity').textContent=selected.capacity;
    }
  }
  spaceGrid.addEventListener('click',event=>{const button=event.target.closest('button[data-space]');if(button)selectSpace(button.dataset.space);});
  function open(){stop();setStep(0);selectSpace('vec');dialog.showModal();}
  document.getElementById('deck').addEventListener('click',event=>{
    const card=event.target.closest('.card[data-group]');
    if(!card || !card.querySelector('.viz-open') || event.target.closest('details,button,a')) return;
    open();
  });
  document.getElementById('deck').addEventListener('keydown',event=>{
    if(event.target.matches('.viz-open') && (event.key==='Enter'||event.key===' ')){event.preventDefault();open();}
  });
  close.addEventListener('click',()=>dialog.close());
  dialog.addEventListener('close',stop);
  play.addEventListener('click',()=>{
    if(timer!==null){stop();return;}
    if(step===3)setStep(0);
    play.textContent='暂停';
    timer=setInterval(()=>{setStep(step+1);if(step===3)stop();},1100);
  });
  previous.addEventListener('click',()=>{stop();setStep(step-1);});
  next.addEventListener('click',()=>{stop();setStep(step+1);});
  range.addEventListener('input',()=>{stop();setStep(range.value);});
  steps.addEventListener('click',event=>{const button=event.target.closest('button[data-step]');if(button){stop();setStep(button.dataset.step);}});
  render();
})();
