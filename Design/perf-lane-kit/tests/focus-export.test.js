import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, readFile, writeFile, rm, readdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {importRun} from '../scripts/import-run.js';
import {importCaptureFiles} from '../src/folder-import.js';
import {findIncore} from '../src/focus-export.js';

test('聚焦导出使用对应快照并启用专题入口', async () => {
 const root=await mkdtemp(join(tmpdir(),'perf-focus-'));
 try {
  await mkdir(join(root,'mock')); await mkdir(join(root,'derived'));
  const snapshot=JSON.parse(await readFile(new URL('../data/example.json',import.meta.url),'utf8'));
  await writeFile(join(root,'mock','lane-data-rendered.json'),JSON.stringify(snapshot));
  const task=snapshot.ranks[0].launches[0];
  await writeFile(join(root,'mock','qk-inline-events.json'),JSON.stringify([{rank:snapshot.ranks[0].rank,taskId:task.taskId,leaf:task.leaf,core:0}]));
  await writeFile(join(root,'mock','qk-pipeline-models.json'),'{}');
  await writeFile(join(root,'derived','pass-analysis.json'),JSON.stringify({cases:[]}));
  await writeFile(join(root,'derived','parallel-lane.json'),'{}');
  for(const s of ['callable','csa-qk-pv']){const dir=join(root,'derived','compile',s);await mkdir(dir,{recursive:true});for(const [name,value] of Object.entries({graph:{nodes:[]},passes:{passes:[]},isa:{},views:{}}))await writeFile(join(dir,name+'.json'),JSON.stringify(value));}
  const imported=await importRun(root);
  assert.equal(imported.focusExport.available,true);
  assert.equal(imported.ranks[0].launches[0].id,snapshot.ranks[0].launches[0].id);
  assert.equal(findIncore(imported,snapshot.ranks[0].rank,task,0).core,0);
  assert.equal(findIncore(imported,'other',task,0),null);
  const files=[];async function walk(dir,relative=''){for(const e of await readdir(dir,{withFileTypes:true})){const path=join(dir,e.name),rel=relative?`${relative}/${e.name}`:e.name;if(e.isDirectory())await walk(path,rel);else files.push({webkitRelativePath:`export/${rel}`,text:()=>readFile(path,'utf8')});}}await walk(root);
  const browser=await importCaptureFiles(files);assert.deepEqual(browser.focusExport,imported.focusExport);
  await writeFile(join(root,'mock','qk-inline-events.json'),'[]');
  await assert.rejects(importRun(root),/缺少有效核内事件/);
 } finally { await rm(root,{recursive:true,force:true}); }
});
