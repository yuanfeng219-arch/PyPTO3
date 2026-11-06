import {readFile,readdir,writeFile,stat} from 'node:fs/promises';
import {resolve,join,basename,relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {adapt} from '../src/data.js';
import {buildCapture} from '../src/dfx-adapter.js';
import {discoverCaptures} from '../src/capture-discovery.js';
import {readFocusExport} from '../src/focus-export.js';
const read=async p=>JSON.parse(await readFile(p,'utf8'));
export async function importRun(runPath){
 const root=resolve(runPath);let dfx=root;
 // The focus export contains a complete rendered snapshot and its associated
 // analysis data. Keep this opt-in to that explicit directory structure.
 let hasFocus=false;
 try { await stat(join(root,'mock','lane-data-rendered.json'));hasFocus=true; }
 catch(e) { if(e.code!=='ENOENT')throw e; }
 if(hasFocus)return await readFocusExport(p=>read(join(root,p)),basename(root));
 try{if((await stat(join(root,'dfx_outputs'))).isDirectory())dfx=join(root,'dfx_outputs');}catch(e){if(e.code!=='ENOENT')throw e;}
 const found=[];async function walk(dir){for(const e of await readdir(dir,{withFileTypes:true})){const p=join(dir,e.name);if(e.isDirectory())await walk(p);else if(e.isFile()&&/^(merged_swimlane_.*|name_map.*|chip_swimlane_records|l2_swimlane_records|deps)\.json$/i.test(e.name))found.push(p);}}await walk(dfx);
 const map=new Map(found.map(p=>['selected/'+relative(dfx,p).replace(/\\/g,'/'),p]));const specs=discoverCaptures([...map.keys()]);
 const ranks=[];for(const spec of specs){const [trace,names,raw,depFile]=await Promise.all([read(map.get(spec.trace)),read(map.get(spec.map)),read(map.get(spec.record)),spec.deps?read(map.get(spec.deps)):null]);ranks.push(buildCapture({rank:spec.rank,source:relative(root,map.get(spec.trace)),trace,map:names,raw,depFile}));}
 return adapt({schemaVersion:'perf-lane.v1',run:basename(root),sourceDirectory:root,ranks,caveats:['时间以各采集最早 AICore 事件为原点；关键路径是依赖图与核上时间包络的分析线索；未匹配的依赖端点不纳入路径。']});
}
if(process.argv[1]&&resolve(process.argv[1])===resolve(fileURLToPath(import.meta.url))){const [input,output]=process.argv.slice(2);if(!input||!output){console.error('用法: node scripts/import-run.js <运行目录或 dfx_outputs> <输出 JSON 路径>');process.exitCode=1;}else try{const doc=await importRun(input);await writeFile(output,JSON.stringify(doc),{flag:'wx'});console.log(`已写入 ${output}: ${doc.ranks.map(r=>`${r.rank} ${r.launches.length} Task / ${r.launches.reduce((n,t)=>n+t.blocks.n,0)} Block`).join('；')}`);}catch(e){console.error(`导入失败：${e.message}`);process.exitCode=1;}}
