/* Read-only engineering import. Imported command strings are never executed.
 * Field/Table/Alert/Collapsible use the existing shadcn-to-PTO adapter. */
window.InvestigationProject=function(A){
 const {esc,button,table,notice}=InvestigationUI;let project=null,sequence=0;
 const sha=async text=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text)))).map(n=>n.toString(16).padStart(2,'0')).join('');
 async function inspect(files,root){
  const entry=files.find(f=>f.path.endsWith('/decode_csa.py')||f.path==='decode_csa.py');
  const source=entry?.text||'';
  const checks=[['源码入口',entry?entry.path:'未识别 decode_csa.py'],['PyPTO / 编译工具链','尚未核对本机安装版本和编译能力'],['正确性基准模块',files.some(f=>/(^|\/)golden(\.py|\/__init__\.py)$/.test(f.path))?'已找到文件；接口与依赖仍需核对':'当前目录没有找到；可能位于外部依赖中'],['输入 / Shape','尚未绑定基线输入、Shape 和随机种子'],['Ascend 设备执行','尚未连接可运行工程的执行服务'],['基线源码版本','尚未绑定历史构建的提交号或内容摘要']];
  const commands=entry?{build:`python ${entry.path} --tp 2 --compile-only --dump-passes`,correctness:`python ${entry.path} --tp 2`,performance:`python ${entry.path} --tp 2 --enable-chip-swimlane 1 --dump-passes`}:{};
  const hashes=await Promise.all(files.map(async f=>({path:f.path,sha256:await sha(f.text)})));
  hashes.sort((a,b)=>a.path.localeCompare(b.path));
  return {root,files,entry:entry?.path,revision:'sha256:'+await sha(JSON.stringify(hashes)),checks,commands,
   flags:{compile:source.includes('--compile-only'),golden:source.includes('golden_fn='),dfx:source.includes('--enable-chip-swimlane')},status:'imported-readonly'};
 }
 function summary(){if(!project)return notice('尚未选择开发工程','可以直接读取仓库中的 DeepSeek 源码，也可以选择另一份本地工程。这里只读取文件，不会运行或修改代码。','info')+button('读取当前 DeepSeek 工程','project-default','btn-sm')+button('选择其他工程目录','project-import','btn-ghost btn-sm');return `<p><strong>${esc(project.root)}</strong> · 已读取 ${project.files.length} 个源码和配置文件</p><details><summary>查看工程内容摘要</summary><code>${esc(project.revision)}</code>${table(['检查项','结果'],project.checks.map(row=>row.map(esc)))}<p>从源码中识别到：编译入口 ${project.flags.compile?'有':'无'} · 正确性基准 ${project.flags.golden?'有':'无'} · Trace 采集入口 ${project.flags.dfx?'有':'无'}。这些结果来自静态扫描，尚未实际执行。</p></details>${notice('还不能在此页面直接运行','需要连接具备 PyPTO、编译工具链和 Ascend 设备的执行服务，并绑定输入与基线源码版本。','info')}`;}
 function present(){A.closeDialog();A.bottom('开发工程检查',summary()+`<div class="inline">${A.activeCase()?.preset?button('定义一次只改一个参数的实验','experiment','btn-sm'):''}${button('换一个工程','project-import','btn-ghost btn-sm')}</div>`);}
 async function preload(){const seq=++sequence;A.toast('正在读取 DeepSeek 工程及源码摘要…');try{const r=await fetch('project-manifest.json');if(!r.ok)throw Error('HTTP '+r.status);const manifest=await r.json();const files=await Promise.all(manifest.files.map(async f=>{const r=await fetch('../../Data/DeepseekV4/'+f.path);if(!r.ok)throw Error('源码缺失：'+f.path);const text=await r.text();if(await sha(text)!==f.sha256)throw Error('源码已变化，请重新生成工程索引：'+f.path);return {path:f.path,text};}));const next=await inspect(files,manifest.root);if(seq!==sequence)return;project=next;project.checks.unshift(['归档产物',`${manifest.passes} 个 Pass 快照 · ${manifest.traces.length} 份 Rank Trace（同一 Run）`]);present();}catch(e){A.dialog('工程导入失败',notice('请检查工程索引',esc(e.message)));}}
 async function importFiles(input){const seq=++sequence;try{const selected=input.filter(f=>/\.(py|toml|json|yaml|yml)$/.test(f.name)&&!/(^|\/)(\.git|\.env|node_modules|credentials|secrets)(\/|$)/i.test(f.webkitRelativePath)&&!/(trace|swimlane)/i.test(f.name));if(!selected.length)throw Error('没有可索引的源码／配置。');if(selected.length>2000||selected.reduce((n,f)=>n+f.size,0)>50e6)throw Error('工程过大；请选择源码子目录（≤ 2,000 文件 / 50 MB）。');const files=await Promise.all(selected.map(async f=>({path:f.webkitRelativePath||f.name,text:await f.text()})));const next=await inspect(files,input[0]?.webkitRelativePath.split('/')[0]||'用户选择的目录');if(seq!==sequence)return;project=next;present();}catch(e){A.dialog('工程导入失败',notice('未运行任何代码',esc(e.message)));}finally{document.getElementById('projectImport').value='';}}
 function binding(){return project;}
 function target(symbol){if(!project)return null;const re=new RegExp('^\\s*'+symbol+'\\s*(?::[^=]+)?=\\s*(\\d+)\\b');for(const file of project.files){if(!file.path.endsWith('.py'))continue;const lines=file.text.split('\n');for(let i=0;i<lines.length;i++){const m=lines[i].match(re);if(m)return {file:file.path,line:i+1,value:Number(m[1]),text:lines[i]};}}return null;}
 return {preload,importFiles,summary,binding,target};
};
