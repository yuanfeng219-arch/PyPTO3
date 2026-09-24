const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ROOT = path.join(__dirname, '..', 'js');
function source(name) { return fs.readFileSync(path.join(ROOT, name), 'utf8'); }
function loadCompilationSchema() {
  const document = { readyState: 'loading', addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; } };
  const window = { location: { search: '' }, addEventListener() {}, document };
  const context = vm.createContext({ window, document, console, URLSearchParams, setTimeout() {}, clearTimeout() {} });
  for (const name of ['ir-pipeline-data.js', 'ir-kernels-data.js', 'ir-compilation-view.js', 'correctness-diagnostic-data.js']) {
    vm.runInContext(source(name), context, { filename: name });
  }
  let history = source('task-history.js').trimEnd();
  const tail = '})();';
  assert.ok(history.endsWith(tail), 'task-history export hook must target the IIFE tail');
  history = history.slice(0, -tail.length) +
    '  window.__compilationSchema = { compilationPageSchema, compilerProfileForRun };\n' + tail;
  vm.runInContext(history, context, { filename: 'task-history.js' });
  return Object.assign(window.__compilationSchema, { window });
}
function run(id, verdict, findings = []) {
  return { id, model: { domains: { compilation: { verdict } }, findings } };
}
function pageFor(schema, item) {
  return schema.compilationPageSchema(item, schema.compilerProfileForRun(item));
}

test('all Compilation pages share the seven-stage validation schema', () => {
  const schema = loadCompilationSchema();
  for (const item of [run('run_105', 'fail'), run('run_106', 'pass'), run('run_109', 'pass')]) {
    const page = pageFor(schema, item);
    assert.deepEqual(JSON.parse(JSON.stringify(page.stages.map(stage => stage.id))),
      ['frontend', 'tensor', 'hierarchy', 'tile', 'kernel', 'memory', 'runtime']);
    assert.ok(page.events.every(event => event.runId === item.id && event.anchor && event.status));
    assert.ok(page.stages.every(stage => page.events.some(event => event.anchor.stage === stage.id)));
  }
});

test('Compilation validation events preserve #105, #109, healthy, and missing-evidence semantics', () => {
  const schema = loadCompilationSchema();
  const r105 = run('run_105', 'fail', [{ domain: 'compilation', summary: 'IR Validation 失败', affectedObjects: [{ kind: 'pass', id: 'LegalizeIndexing' }], evidence: ['动态 index 约束无法通过'] }]);
  const page105 = pageFor(schema, r105);
  assert.equal(page105.events.find(event => event.anchor.id === 'LegalizeIndexing').status, 'error');
  assert.equal(page105.events.find(event => event.anchor.id === 'AllocateMemory').status, 'not_run');
  assert.equal(page105.events.find(event => event.anchor.id === 'Codegen').status, 'not_run');

  const page109 = pageFor(schema, run('run_109', 'pass'));
  assert.equal(page109.events.find(event => event.anchor.id === 'ExpandMixedKernel').status, 'first_divergence');
  assert.equal(page109.events.find(event => event.anchor.id === 'InjectGMPipeBuffer').status, 'error');
  assert.equal(page109.events.find(event => event.anchor.id === 'device-execution').status, 'not_run');

  const page106 = pageFor(schema, run('run_106', 'pass'));
  assert.equal(page106.events.filter(event => event.type === 'numerical' && event.status === 'pass').length, 42);
  assert.equal(page106.events.find(event => event.anchor.id === 'output-validation').status, 'error');
  assert.equal(page106.events.find(event => event.anchor.id === 'device-execution').status, 'error');

  const unknownPage = pageFor(schema, run('unknown', 'unknown'));
  assert.ok(unknownPage.events.every(event => event.status === 'not_collected'));
});

test('the shared Compilation renderer always emits the three schema layers', () => {
  const schema = loadCompilationSchema();
  const host = { dataset: {}, innerHTML: '', addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; } };
  const r105 = run('run_105', 'fail', [{ domain: 'compilation', summary: 'IR Validation 失败', affectedObjects: [{ kind: 'pass', id: 'LegalizeIndexing' }], evidence: [] }]);
  schema.window.PTO_RUN_CONTEXT = { compilationSchema: pageFor(schema, r105), runId: r105.id };
  assert.equal(schema.window.PTO_COMPILATION.render(host), true);
  assert.match(host.innerHTML, /Investigation Handoff/);
  assert.match(host.innerHTML, /Validation Flow/);
  assert.match(host.innerHTML, /Selected Validation \/ Pass Inspector/);
  assert.match(host.innerHTML, /LegalizeIndexing · ERROR/);

  const r109 = run('run_109', 'pass');
  schema.window.PTO_RUN_CONTEXT = { compilationSchema: pageFor(schema, r109), runId: r109.id };
  schema.window.PTO_COMPILATION.setNumericalContext(null, null);
  assert.match(host.innerHTML, /ExpandMixedKernel · FIRST DIVERGENCE/);
  assert.match(host.innerHTML, /data-kc-original-validation/);
});
