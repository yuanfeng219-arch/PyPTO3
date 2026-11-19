import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url));
const repo=path.resolve(here,'../..');
const fixture=JSON.parse(fs.readFileSync(path.join(repo,'tests/fixtures/model-architecture-3d-deck/source-baseline.json'),'utf8'));
const js=fs.readFileSync(path.join(repo,'patterns/model-architecture-3d-deck/pattern.js'),'utf8');
const css=fs.readFileSync(path.join(repo,'patterns/model-architecture-3d-deck/pattern.css'));
const html=fs.readFileSync(path.join(repo,'patterns/model-architecture-3d-deck/pattern.html'));
const start=js.indexOf(fixture.coreRendererBoundary.start),end=js.indexOf(fixture.coreRendererBoundary.end);
const hash=value=>crypto.createHash('sha256').update(value).digest('hex');

assert.ok(start>=0&&end>start,'Frozen Core renderer boundaries must still exist');
assert.equal(hash(js.slice(start,end)),fixture.coreRendererSha256,'Core Layer renderer changed without an approved baseline update');
assert.equal(hash(css),fixture.patternCssSha256,'Original Layer CSS changed without an approved baseline update');
assert.equal(hash(html),fixture.patternHtmlSha256,'Independent Layer preview changed without an approved baseline update');

console.log('model-architecture-3d-deck source integrity: 3 frozen assets unchanged');
