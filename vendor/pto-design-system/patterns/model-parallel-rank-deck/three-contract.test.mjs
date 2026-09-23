import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

globalThis.window=globalThis;
await import('./pattern.js');

const api=globalThis.PtoModelParallelRankDeck;
const here=path.dirname(fileURLToPath(import.meta.url));
const html=fs.readFileSync(path.join(here,'pattern.html'),'utf8');
const json=JSON.parse(fs.readFileSync(path.join(here,'pattern.json'),'utf8'));
const spec=fs.readFileSync(path.join(here,'THREEJS_RANK_PACKING_SPEC.md'),'utf8');
const runtime=path.join(here,'vendor/three.module.min.js');
const core=path.join(here,'vendor/three.core.min.js');

assert.equal(api.sampleSvgPath('M0 0L10 0').length,2,'Straight source paths must preserve both endpoints');
assert.equal(api.sampleSvgPath('M0 0C0 10 10 10 10 0').length,6,'Cubic source paths must be sampled, not collapsed');
assert.match(html,/import \* as THREE from '\.\/vendor\/three\.module\.min\.js'/);
assert.match(html,/Three\.js Rank containers/);
assert.ok(fs.statSync(runtime).size>300_000,'Pinned Three.js module is missing or truncated');
assert.ok(fs.statSync(core).size>300_000,'Pinned Three.js core is missing or truncated');
assert.equal(json.source.spec,'patterns/model-parallel-rank-deck/THREEJS_RANK_PACKING_SPEC.md');
assert.match(spec,/Do not replace a Layer with a glyph/);
assert.match(spec,/Creating or cloning one Three\.js Group per Rank is forbidden/);
assert.match(spec,/No continuous render loop while the scene is idle/);

console.log('model-parallel-rank-deck Three.js contract: 10 assertions passed');
