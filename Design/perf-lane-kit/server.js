import {createServer} from 'node:http';
import {readFile, mkdtemp, rm} from 'node:fs/promises';
import {rmSync} from 'node:fs';
import {resolve,extname,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {tmpdir} from 'node:os';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {importRun} from './scripts/import-run.js';
const root=resolve(fileURLToPath(new URL('.',import.meta.url))),referenceRoot=resolve(root,'../device-perf-lane-20261026');
const args=process.argv.slice(2);let dataset,runRoot=args[1];
try{
 if(args.length&&(args.length!==2||args[0]!=='--run'))throw Error('用法: node server.js [--run 运行目录或 ZIP]');
 if(runRoot&&extname(runRoot).toLowerCase()==='.zip'){
  const run=promisify(execFile),zip=resolve(runRoot);
  const {stdout}=await run('tar',['-tvf',zip],{maxBuffer:16*1024*1024});
  const lines=stdout.split(/\r?\n/).filter(Boolean);
  if(!lines.length||lines.some(line=>!/^[-d]/.test(line)))throw Error('ZIP 含不支持的文件类型');
  const listing=await run('tar',['-tf',zip],{maxBuffer:16*1024*1024});
  const entries=listing.stdout.split(/\r?\n/).filter(Boolean).map(s=>s.replace(/\\/g,'/'));
  if(entries.some(s=>s.startsWith('/')||/^[A-Za-z]:/.test(s)||s.split('/').includes('..')))throw Error('ZIP 包含不安全的路径');
  const temp=await mkdtemp(resolve(tmpdir(),'perf-lane-export-'));
  try{await run('tar',['-xf',zip,'-C',temp]);runRoot=resolve(temp,'export-focus-data');}
  catch(e){await rm(temp,{recursive:true,force:true});throw e;}
  process.on('exit',()=>{try{rmSync(temp,{recursive:true,force:true});}catch{}});
 }
 dataset=runRoot?await importRun(runRoot):JSON.parse(await readFile(resolve(root,'data/example.json'),'utf8'));
}catch(e){console.error(`启动失败：${e.message}`);process.exit(1);}
const payload=JSON.stringify(dataset),mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8'};
const bridge=`window.addEventListener('message', e=>{if(e.origin!==location.origin||e.source!==window.parent||e.data?.type!=='perf-lane-drilldown')return;const m=e.data;if(m.scene!=='deepseek')return;if(D.ranks.some(r=>r.rank===m.rank)){$('rank').value=m.rank;loadRank(m.rank);}const t=R.launches.find(x=>x.id===m.task&&window.QK_RESIDENCY.isTarget(x));if(t)select(t.id,Number.isFinite(m.core)?m.core:null,false,true);});\n`;
createServer(async(req,res)=>{try{const url=new URL(req.url,'http://localhost'),pathname=decodeURIComponent(url.pathname);
 if(pathname==='/drilldown'){let html=await readFile(resolve(referenceRoot,'index.html'),'utf8');const end='  loadRank(D.ranks[0].rank);';if(!html.includes(end))throw Error('reference bridge marker missing');const at=html.lastIndexOf(end);html=(html.slice(0,at)+end+'\n'+bridge+html.slice(at+end.length)).replace('<head>','<head><base href="/reference/">');res.writeHead(200,{'Content-Type':mime['.html']});res.end(html);return;}
 if(pathname.startsWith('/reference/')){const ref=resolve(referenceRoot,'.'+pathname.slice('/reference'.length));if(ref!==referenceRoot&&!ref.startsWith(referenceRoot+sep))throw Error('invalid path');const file=ref===referenceRoot?resolve(referenceRoot,'index.html'):ref;const contents=await readFile(file);res.writeHead(200,{'Content-Type':mime[extname(file)]||'application/octet-stream'});res.end(contents);return;}
 if(pathname==='/api/data'){res.writeHead(200,{'Content-Type':mime['.json'],'Cache-Control':'no-store'});res.end(payload);return;}
 const path=resolve(root,'.'+pathname);if(path!==root&&!path.startsWith(root+sep))throw Error('invalid path');const file=path===root?resolve(root,'index.html'):path;const contents=await readFile(file);res.writeHead(200,{'Content-Type':mime[extname(file)]||'application/octet-stream'});res.end(contents);
 }catch{res.writeHead(404);res.end('Not found');}}).listen(0,'127.0.0.1',function(){console.log(`http://127.0.0.1:${this.address().port}/`);});
