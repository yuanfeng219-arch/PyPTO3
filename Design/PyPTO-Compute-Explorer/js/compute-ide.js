(function(){'use strict';

const editor=document.getElementById('dslEditor'), bar=document.getElementById('sourceRegionBar'), scroller=document.querySelector('.kf-main-body');
  const pagedAttentionPyptoFocusMeta = {
    constants: { label: '模块常量与静态断言', lines: '9–65', detail: 'BATCH / HEAD_DIM / BLOCK_SIZE / STACK_* 等编译期常量与 8 条 assert 自检' },
    signature: { label: '函数签名与张量契约', lines: '67–108', detail: '27 个 Tensor / TaskID 入参，导出 out，并把分页缓存 reshape 成 BSND 视图' },
    spmd: { label: 'SPMD 入口与生产者依赖', lines: '109–126', detail: '24 个 SPMD block，显式声明 7 个 Phase-0 生产者的 TaskId 依赖' },
    kvphase: { label: 'Phase 0 · K/V 归一化与 Cache 写回', lines: '127–213', detail: '前 32 个 AIV lane：K 做 RMSNorm + RoPE 写 key cache，V 乘 inv_rms 写 value cache' },
    qphase: { label: 'Phase 0 · Q 归一化与 RoPE', lines: '215–251', detail: '每组 GROUP 个 query head 做 RMSNorm + RoPE，按 BSND 行写回 q2d' },
    barrier: { label: 'Phase-0 发布与全核 Barrier', lines: '253–259', detail: 'cacheinvalid → fence → syncall(MIX) → cacheinvalid，保证 PA 读到已发布的 Q/K/V' },
    taskloop: { label: 'Attention 任务循环与 Q 装载', lines: '261–277', detail: '每个 (batch, kv_head) 一个 task，推导 page / stack 数并把 Q tile 装进 L1' },
    qk: { label: 'QK 生产 · Cube', lines: '278–362', detail: '按 STACK_PAGES 逐页 load K、matmul 出 score 写入 transfer，含 1/2/3 页末栈 tail 分支' },
    pv: { label: 'PV 生产 · Cube', lines: '364–724', detail: '一次预取整栈 V 并 ping-pong P，按页数分 1/2/3/4 页 tail 分支做 matmul_acc' },
    softmax: { label: '在线 Softmax 与 FIFO 更新 · Vector', lines: '726–862', detail: '双 AIV lane 上 scale / mask / row_max / exp / row_sum，用 pending FIFO 做三态 rescale' },
    out: { label: '输出归一化与写回', lines: '864–875', detail: 'o / l_sum 归一化并转 BF16 写回 out2d，返回本 kernel 的 attn_tid' },
  };

  const PY_KEYWORDS = new Set([
    'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def',
    'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if',
    'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise',
    'return', 'try', 'while', 'with', 'yield', 'match', 'case'
  ]);

  // 内置函数 / 类型 / 异常。单独给一个颜色，和普通调用区分开。
  const PY_BUILTINS = new Set([
    'abs', 'aiter', 'all', 'anext', 'any', 'ascii', 'bin', 'bool', 'breakpoint',
    'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
    'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter',
    'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help',
    'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list',
    'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open',
    'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set',
    'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple',
    'type', 'vars', 'zip',
    'ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException',
    'Exception', 'ImportError', 'IndexError', 'KeyError', 'KeyboardInterrupt',
    'MemoryError', 'NameError', 'NotImplementedError', 'OSError', 'OverflowError',
    'RuntimeError', 'StopIteration', 'TypeError', 'ValueError', 'ZeroDivisionError',
  ]);

  function highlightPythonLines(source) {
    const esc = (s) => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    const span = (cls, s) => `<span class="${cls}">${esc(s)}</span>`;
    const lines = source.split('\n');
    const out = [];
    let triple = null; // { quote, cls } — 跨行中的三引号字符串

    for (let li = 0; li < lines.length; li++) {
      const line = lines[li];
      let html = '';
      let i = 0;

      // 同类相邻 token 合并成一个 span。decode_layer.py 有 1768 行，逐字符各出一个
      // span 会把 DOM 体积翻好几倍；plain 与 token 两个缓冲互斥，谁在用另一个必为空。
      let plain = '';
      let tokCls = null;
      let tokBuf = '';
      const flushPlain = () => { if (plain) { html += esc(plain); plain = ''; } };
      const flushTok = () => { if (tokBuf) { html += span(tokCls, tokBuf); tokCls = null; tokBuf = ''; } };
      const pushTok = (cls, s) => {
        if (tokCls !== cls) { flushPlain(); flushTok(); tokCls = cls; }
        tokBuf += s;
      };
      const pushPlain = (s) => { flushTok(); plain += s; };

      if (triple) {
        const close = line.indexOf(triple.quote);
        if (close === -1) { out.push(span(triple.cls, line)); continue; }
        html += span(triple.cls, line.slice(0, close + 3));
        i = close + 3;
        triple = null;
      }

      let expectName = null; // 'func' | 'class' — next identifier is a definition name
      let prev = '';         // 上一个有意义的字符，用于判断 x.y 是不是属性访问

      while (i < line.length) {
        const rest = line.slice(i);
        const ch = line[i];

        // comment
        if (ch === '#') { pushTok('tok-com', line.slice(i)); i = line.length; break; }

        // 字符串（可带 r/b/u/f 前缀）。三引号且顶在视觉行首 → 当 docstring 单独上色。
        const sm = /^([rRbBuUfF]{0,2})("""|'''|"|')/.exec(rest);
        if (sm) {
          const prefixLen = sm[1].length;
          const q = sm[2];
          const startQuote = i + prefixLen;
          const tokenStart = i;
          const cls = q.length === 3 && line.slice(0, tokenStart).trim() === '' ? 'tok-doc' : 'tok-str';
          let text;
          if (q.length === 3) {
            const after = line.indexOf(q, startQuote + 3);
            if (after === -1) {
              pushTok(cls, line.slice(tokenStart));
              triple = { quote: q, cls };
              i = line.length;
              break;
            }
            text = line.slice(tokenStart, after + 3);
            i = after + 3;
          } else {
            let j = startQuote + 1;
            while (j < line.length) {
              if (line[j] === '\\') { j += 2; continue; }
              if (line[j] === q) { j++; break; }
              j++;
            }
            text = line.slice(tokenStart, j);
            i = j;
          }
          // f-string：把 {...} 里的内容单独上色（只做一层，不解析格式串与嵌套花括号）
          const bodyFrom = prefixLen + q.length;
          if (/[fF]/.test(sm[1]) && text.length >= bodyFrom + q.length && text.endsWith(q)) {
            pushTok(cls, text.slice(0, bodyFrom));
            const body = text.slice(bodyFrom, text.length - q.length);
            const re = /\{[^{}]*\}/g;
            let at = 0;
            let brace;
            while ((brace = re.exec(body))) {
              if (brace.index > at) pushTok(cls, body.slice(at, brace.index));
              pushTok('tok-interp', brace[0]);
              at = brace.index + brace[0].length;
            }
            if (at < body.length) pushTok(cls, body.slice(at));
            pushTok(cls, text.slice(text.length - q.length));
          } else {
            pushTok(cls, text);
          }
          prev = text.slice(-1);
          continue;
        }

        // number
        const nm = /^(0[xXoObB][0-9a-fA-F_]+|(?:\d[\d_]*\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[jJ]?)/.exec(rest);
        if (nm && /\d/.test(nm[0])) { pushTok('tok-num', nm[0]); prev = nm[0].slice(-1); i += nm[0].length; continue; }

        // decorator (only at the visual start of a line)
        if (ch === '@') {
          const dm = /^@[A-Za-z_][\w.]*/.exec(rest);
          if (dm && line.slice(0, i).trim() === '') { pushTok('tok-dec', dm[0]); prev = dm[0].slice(-1); i += dm[0].length; continue; }
        }

        // identifier / keyword / 属性
        const im = /^[A-Za-z_]\w*/.exec(rest);
        if (im) {
          const word = im[0];
          const isCall = /^\s*\(/.test(line.slice(i + word.length));
          const isAttr = prev === '.';
          let cls = null;
          if (expectName) { cls = expectName === 'class' ? 'tok-cls' : 'tok-fn'; expectName = null; }
          else if (word === 'def') { cls = 'tok-kw'; expectName = 'func'; }
          else if (word === 'class') { cls = 'tok-kw'; expectName = 'class'; }
          else if (isAttr) cls = isCall ? 'tok-fn' : 'tok-prop';
          else if (PY_KEYWORDS.has(word)) cls = 'tok-kw';
          else if (word === 'True' || word === 'False' || word === 'None') cls = 'tok-const';
          else if (word === 'self' || word === 'cls') cls = 'tok-self';
          else if (PY_BUILTINS.has(word)) cls = 'tok-builtin';
          else if (isCall) cls = 'tok-fn';
          if (cls) pushTok(cls, word); else pushPlain(word);
          prev = word.slice(-1);
          i += word.length;
          continue;
        }

        // 运算符（相邻同类会合并成一个 span，所以不用穷举 **= / >>= 这类组合）
        if (/^(?::=|[-+*/%@&|^~<>=!])/.test(rest)) { pushTok('tok-op', ch); prev = ch; i++; continue; }
        // 括号与分隔符（. 不在此列，属性链保持干净）
        if (/[()[\]{},:;]/.test(ch)) { pushTok('tok-punc', ch); prev = ch; i++; continue; }

        pushPlain(ch);
        prev = ch;
        i++;
      }
      flushTok();
      flushPlain();
      out.push(html);
    }
    return out;
  }



const csaRegions = {
 config:{label:'TP 与 Shape 配置',lines:'17–154',detail:'TP=2；本地 query 与 gathered KV 两条动态轴'},
 contract:{label:'decode_csa · 输入与窗口',lines:'157–215',detail:'Rank-local 输入、Cache 状态和 DistributedTensor'},
 pre:{label:'HC Pre / RMSNorm / RoPE',lines:'216–274',detail:'本地 token 预处理与 rope_tid'},
 gather:{label:'Token AllGather',lines:'275–288',detail:'本地 x_normed_t 汇聚为 rank-major x_normed_full'},
 projection:{label:'Q / KV Projection',lines:'290–321',detail:'Q 读取本地行；KV 读取整组行'},
 cache:{label:'Cache / Compressor / Indexer',lines:'322–381',detail:'三类 Cache、两路压缩状态和显式 TaskID 依赖'},
 attention:{label:'Sparse Attention',lines:'383–393',detail:'本地 Q 查询 Original / Compressed KV 与 Top-k'},
 publish:{label:'Merge / Pack / Publish',lines:'395–479',detail:'按 Attention Group 向目标 Rank 的窗口发布'},
 output:{label:'All-to-All / O / ReduceScatter',lines:'481–501',detail:'收齐本地 Group 的整组 token，再按 token owner 归约'},
 launch:{label:'l3_decode_csa · 多设备启动',lines:'624–751',detail:'窗口分配、rank 切片和 device=rank'},
 fixtures:{label:'TP1 / 数据构造 / Reference',lines:'754–2079',detail:'单卡路径、分布式输入构造与 Golden / CLI'}
};
// 两个主文件保留人工编写的 Source Region；其余文件由 project-files.js 依据真实 def / class 自动分区。
const regionMeta={
 'paged_attention_pypto.py':pagedAttentionPyptoFocusMeta,
 'decode_csa.py':csaRegions
};
const files={};
(window.PtoProjectFiles?.files||[]).forEach(file=>{
  files[file.name]={
    id:file.id,name:file.name,scene:file.scene,role:file.role,sourcePath:file.sourcePath,origin:file.origin,
    source:file.content,
    regions:regionMeta[file.name]||file.regions,
    description:file.scene==='csa'
      ?'DeepSeek V4 CSA · '+file.name+'（'+file.role+'）· 共 '+file.lineCount+' 行'
      :'Paged Attention · Task / Block / AIV 的映射与计算关系。'
  };
});
// 注册表缺失时的最小兜底，保证两个原有场景仍可打开。
if(!files['paged_attention_pypto.py'])files['paged_attention_pypto.py']={scene:'paged-attention',source:window.PTO_PAGED_ATTENTION_PYPTO_SOURCE,regions:pagedAttentionPyptoFocusMeta,description:'Paged Attention · Task / Block / AIV 的映射与计算关系。'};
if(!files['decode_csa.py'])files['decode_csa.py']={scene:'csa',source:window.PTO_CSA_SOURCE.text,regions:csaRegions,description:'DeepSeek V4 CSA · TP=2 的数据分布、跨 Rank 通信与任务依赖。'};
let activeFile='paged_attention_pypto.py',entries=[],selected=null,expanded=false,programmatic=false,lastRegion;
const opened=[activeFile];
const sessions=new Map();
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function focusForLine(n){return entries.filter(([,v])=>n>=Number(v.lines.split('–')[0])).at(-1)?.[0]||null;}
function current(){const rows=editor.children,top=scroller.getBoundingClientRect().top+bar.offsetHeight+8;let lo=0,hi=rows.length-1,at=0;while(lo<=hi){const mid=(lo+hi)>>1;if(rows[mid].getBoundingClientRect().bottom>top){at=mid;hi=mid-1;}else lo=mid+1;}return rows[at]?.dataset.pagedAttentionFocus||null;}
function renderBar(){const active=selected||current(),index=entries.findIndex(([k])=>k===active),item=index>=0?entries[index][1]:(entries.length?{label:'文件头',lines:'1–'+(Number(entries[0][1].lines.split('–')[0])-1),detail:'许可证声明与模块说明'}:{label:'全量源码',lines:'1–'+files[activeFile].source.split('\n').length,detail:'该文件没有模块级 def / class 索引，按行浏览'});lastRegion=active;bar.hidden=false;
 bar.innerHTML='<div class="kf-source-region__bar"><button type="button" class="kf-source-region__head" data-source-region-toggle aria-expanded="'+expanded+'"><span class="kf-source-region__pos">'+(entries.length?(index>=0?index+1:'—')+'/'+entries.length:'—')+'</span><b>'+item.label+'</b><i>'+item.lines+'</i><small>'+item.detail+'</small><em>'+(expanded?'收起':'全部')+'</em></button><button type="button" class="btn btn-sm" data-visualization2-open aria-controls="'+(files[activeFile].scene==='csa'?'csaExplorer':'visualization2')+'">Compute Explorer</button></div><div class="kf-source-region__list" '+(expanded?'':'hidden')+'>'+entries.map(([key,v],i)=>'<button type="button" class="'+(key===active?'is-active':'')+'" data-source-region="'+key+'"><span>'+String(i+1).padStart(2,'0')+'</span><b>'+v.label+'</b><i>'+v.lines+'</i><small>'+v.detail+'</small></button>').join('')+'</div>';
}
function showLine(start,end=start){programmatic=true;expanded=false;selected=focusForLine(start);renderBar();const target=editor.children[start-1];if(target){scroller.scrollTo({top:scroller.scrollTop+target.getBoundingClientRect().top-scroller.getBoundingClientRect().top-bar.offsetHeight,behavior:'instant'});editor.querySelectorAll('[data-paged-attention-line]').forEach((el,i)=>el.classList.toggle('is-paged-attention-line-active',i+1>=start&&i+1<=end));}setTimeout(()=>{programmatic=false;},150);}
// --- 跨文件符号链接 ---------------------------------------------------------
// 只根据 project-files.js 的静态 import → 定义行索引建立链接；解析不到的名字保持普通文本。
const linkCache=new Map();
let currentLinks=null;
function linkContext(name){
  if(linkCache.has(name))return linkCache.get(name);
  const links=window.PtoProjectFiles?.linksFor?.(files[name]?.id);
  const ctx={re:null,symbols:{},modules:{},importRanges:[],list:[],index:new Map()};
  if(links){
    ctx.symbols=links.symbols||{};ctx.modules=links.modules||{};ctx.importRanges=links.importLines||[];
    const names=[...new Set([...Object.keys(ctx.symbols),...Object.keys(ctx.modules)])].sort((a,b)=>b.length-a.length);
    if(names.length)ctx.re=new RegExp('\\b(?:'+names.map(n=>n.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|')+')\\b','g');
  }
  linkCache.set(name,ctx);return ctx;
}
const PROTECTED_TOKENS=new Set(['tok-str','tok-doc','tok-com','tok-interp']);
function linkifyLine(html,lineNo,ctx){
  if(!ctx.re)return html;
  const inImport=ctx.importRanges.some(range=>lineNo>=range[0]&&lineNo<=range[1]);
  const swap=text=>text.replace(ctx.re,match=>{
    const target=ctx.symbols[match]||(inImport?ctx.modules[match]:null);
    if(!target)return match;
    const key=target.file+':'+target.line;
    if(!ctx.index.has(key)){ctx.index.set(key,ctx.list.length);ctx.list.push({file:target.file,line:target.line});}
    const label=(target.file.split('/')[1]||target.file)+' : '+target.line;
    return '<button type="button" class="pf-link" data-pf-link="'+ctx.index.get(key)+'" title="跳转 · '+escape(label)+'">'+match+'</button>';
  });
  const parts=html.split(/(<span class="tok-[a-z]+">|<\/span>)/);
  let token=null;
  let out='';
  for(const part of parts){
    if(part.startsWith('<span class="tok-')){token=/tok-([a-z]+)/.exec(part)?.[1]||null;out+=part;continue;}
    if(part==='</span>'){token=null;out+=part;continue;}
    out+=token&&PROTECTED_TOKENS.has('tok-'+token)?part:swap(part);
  }
  return out;
}
function renderTabs(){
  const host=document.getElementById('tabs');
  if(!host)return;
  host.innerHTML=opened.map(file=>'<button data-source-file="'+escape(file)+'" title="'+escape(files[file]?.sourcePath||file)+'" class="'+(file===activeFile?'is-active':'')+'">'+escape(file)+'</button>').join('');
}
function openFile(name){if(!files[name])return;if(editor.children.length)sessions.set(activeFile,{scroll:scroller.scrollTop,selected});window.PtoCsaExplorer?.close();if(document.querySelector('#ideWorkarea').dataset.visualization2)window.PtoVisualization2?.close();activeFile=name;entries=Object.entries(files[name].regions||{});selected=sessions.get(name)?.selected||null;expanded=false;
 if(!opened.includes(name))opened.push(name);
 const links=linkContext(name);currentLinks=links;
 const lines=files[name].source.split('\n'),colored=highlightPythonLines(files[name].source);
 editor.innerHTML=lines.map((line,i)=>'<div data-paged-attention-line="'+(i+1)+'" data-paged-attention-focus="'+(focusForLine(i+1)||'')+'"><i>'+(i+1)+'</i><code style="--kf-code-indent:'+(line.match(/^\s*/)||[''])[0].length+'ch">'+(colored[i]?linkifyLine(colored[i],i+1,links):' ')+'</code></div>').join('');
 editor.setAttribute('aria-label',name+' 全量源码');document.querySelector('.kf-source-stage').setAttribute('aria-label',name+' 全量源码');
 document.querySelectorAll('[data-current-file]').forEach(el=>el.textContent=name);document.querySelectorAll('[data-file]').forEach(el=>el.classList.toggle('is-selected',el.dataset.file===name));
 renderTabs();
 document.getElementById('inspector').innerHTML='<div class="kf-inspector-card"><b>'+escape(name)+'</b><p>'+escape(files[name].description)+'</p>'+(files[name].sourcePath?'<p class="pf-source-meta">'+escape(files[name].sourcePath)+'</p>':'')+'<button class="btn btn-sm" type="button" data-visualization2-open>Compute Explorer</button></div>';
 document.body.dataset.computeScene=files[name].scene;renderBar();scroller.scrollTo({top:sessions.get(name)?.scroll||0,behavior:'instant'});document.dispatchEvent(new CustomEvent('compute-file-change',{detail:{file:name,scene:files[name].scene}}));}
bar.addEventListener('click',e=>{if(e.target.closest('[data-source-region-toggle]')){expanded=!expanded;renderBar();return;}const b=e.target.closest('[data-source-region]');if(b){const[a,z]=files[activeFile].regions[b.dataset.sourceRegion].lines.split('–').map(Number);showLine(a,z);}});
scroller.addEventListener('scroll',()=>{if(programmatic)return;selected=null;if(current()!==lastRegion)renderBar();},{passive:true});
editor.addEventListener('click',e=>{const link=e.target.closest('[data-pf-link]');if(link&&currentLinks){const target=currentLinks.list[Number(link.dataset.pfLink)];if(target){e.stopPropagation();openSource({file:target.file,start:target.line,end:target.line});return;}}
 const row=e.target.closest('[data-paged-attention-line]');if(!row)return;selected=row.dataset.pagedAttentionFocus;editor.querySelectorAll('[data-paged-attention-focus]').forEach(el=>el.classList.toggle('is-paged-attention-line-active',el.dataset.pagedAttentionFocus===selected));renderBar();});
document.addEventListener('click',e=>{const f=e.target.closest('[data-file],[data-source-file]');if(f)openFile(f.dataset.file||f.dataset.sourceFile);const folder=e.target.closest('[data-tree-toggle]');if(folder){const group=document.querySelector('[data-tree-group="'+folder.dataset.treeToggle+'"]');if(group){group.hidden=!group.hidden;folder.setAttribute('aria-expanded',String(!group.hidden));}}});
// --- 文件级源码导航 ---------------------------------------------------------
function resolveEntry(key){
  if(!key)return files[activeFile]?files[activeFile]:null;
  if(files[key])return files[key];
  const entry=window.PtoProjectFiles?.get?.(key);
  return entry&&files[entry.name]?files[entry.name]:null;
}
// 打开源码前先退出可视化覆盖层，避免跳转结果被 Explorer 遮住。
function leaveVisualization(){
  if(window.PtoCsaExplorer?.getState?.().open)window.PtoCsaExplorer.close();
  if(window.PtoVisualization2?.getState?.().open)window.PtoVisualization2.close();
}
function openSource(options){
  const request=typeof options==='string'?{file:options}:(options||{});
  const entry=resolveEntry(request.file||activeFile);
  if(!entry)throw new Error('未注册的源码文件 '+request.file);
  if(entry.name!==activeFile)openFile(entry.name);else leaveVisualization();
  const start=Math.max(1,Number(request.start)||1),end=Math.max(start,Number(request.end)||start);
  showLine(start,end);editor.focus({preventScroll:true});
  return entry.name;
}
function showSourceFile(name,start,end=start){return openSource({file:name,start,end});}
window.PtoComputeIDE=Object.freeze({
  getActiveFile:()=>activeFile,
  getScene:()=>files[activeFile].scene,
  openFile,
  openSource,
  showSource(start,end){return openSource({file:activeFile,start,end});},
  showSourceFile,
  listFiles:()=>Object.keys(files)
});
openFile(activeFile);
})();
