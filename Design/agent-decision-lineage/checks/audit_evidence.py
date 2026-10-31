"""Read-only local evidence audit. Run from any directory; optionally check HTTP links.
Usage: python3 checks/audit_evidence.py [http://127.0.0.1:8765/]
"""
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parents[1] / 'Data' / 'softmax 新研pto pro agent产物数据'
results = []

def check(name, ok):
    results.append({'name': name, 'passed': bool(ok)})

def bundle(name):
    return json.loads((ROOT / name).read_text().split('=', 1)[1].strip().rstrip(';'))

model = json.loads((ROOT / 'lineage-data.json').read_text())
check('JSON / JS model parity', model == bundle('lineage-data.js'))
ids = [n['id'] for n in model['nodes']]
check('unique node IDs', len(ids) == len(set(ids)))
check('valid edge endpoints', all(e['source'] in ids and e['target'] in ids for e in model['edges']))
check('unique source IDs', len({s['id'] for s in model['sources']}) == len(model['sources']))
for s in model['sources']:
    p = ROOT / s['path']
    check('snapshot hash: ' + s['file'], p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == s['sha256'])
    check('original parity: ' + s['file'], (ORIGINAL / s['file']).read_bytes() == p.read_bytes())
    if len(sys.argv) > 1:
        with urllib.request.urlopen(sys.argv[1].rstrip('/') + '/' + s['path']) as response:
            check('HTTP source: ' + s['file'], response.status == 200 and response.read() == p.read_bytes())
source_ids = {s['id'] for s in model['sources']}
check('node evidence references', all(r['sourceId'] in source_ids and r.get('location') for n in model['nodes'] for r in n['evidenceRefs']))
check('edge evidence references', all(r['sourceId'] in source_ids and r.get('location') for e in model['edges'] for r in e['evidenceRefs']))
exp = bundle('experiment-data.js')
report = (ROOT / 'evidence/PERFORMANCE_REPORT.md').read_text()
for label, section in [('baseline', report.split('## 3. Baseline')[1].split('### Baseline')[0]), ('final', report.split('## 5. Final')[1].split('Aggregate golden')[0])]:
    rows = [line for line in section.splitlines() if line.startswith('| p0_')]
    for row in rows:
        cells = [c.strip() for c in row.strip('|').split('|')]
        case = next(c for c in exp['cases'] if c['id'] == cells[0])
        check(label + ' formal values / samples: ' + cells[0], case[label]['us'] == float(cells[3]) and case[label]['samples'] == json.loads(cells[4]))
        check(label + ' aggregation: ' + cells[0], sorted(case[label]['samples'])[1] == case[label]['us'])
check('baseline code snapshot parity', exp['codeBefore'] == (ROOT / 'evidence/test_softmax.py.bak_stage5').read_text())
check('final code snapshot parity', exp['codeAfter'] == (ROOT / 'evidence/test_softmax.py').read_text())
for name, contents in bundle('decision-evidence.js').items():
    check('design bundle parity: ' + name, contents == (ROOT / 'evidence' / name).read_text())
correctness = bundle('audit-evidence.js')['correctness']
check('all eight reported correctness cases', len(correctness) == 8 and all(r['status'] == 'PASS' and r['matched_ratio'] == '1.0' for r in correctness))
for row in correctness:
    check('correctness report row: ' + row['case'], bool(re.search(r'\|\s*' + re.escape(row['case']) + r'\s*\|\s*' + re.escape(row['shape']) + r'\s*\|\s*1.0\s*\|\s*' + re.escape(row['max_abs_error']), report)))
for metric in exp['profiling']:
    check('PMU report: ' + metric['metric'], metric['baseline'] in report and metric['final'] in report)
layout = (ROOT / 'lineage-layout.js').read_text()
items = re.findall(r"\['(e[0-9]+|[dxi]-[^']*)','", layout)
branches = re.findall(r"\{id:'((?:d|x)-[^']+)'", layout)
shown = items + branches
check('unique displayed node identities', len(shown) == len(set(shown)))
check('displayed nodes exist', all(i in ids for i in shown))
journey = (ROOT / 'experiment-inspector.js').read_text()
check('PASS maps x-verify', "['8/8 PASS','x-verify']" in journey)
check('finding / hypothesis separate', "['重复 update_mask','f-mask']" in journey and "['优化假设','h-mask']" in journey)
check('Empty-tail component edge', any(e['source'] == 'x-mask' and e['target'] == 'x-tail' and e['type'] == 'includes' for e in model['edges']))
output = {'checks': len(results), 'passed': all(r['passed'] for r in results), 'results': results}
(ROOT / 'previews/evidence-audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
print(json.dumps({'checks': len(results), 'passed': output['passed'], 'failures': [r for r in results if not r['passed']]}, ensure_ascii=False))
sys.exit(0 if output['passed'] else 1)
