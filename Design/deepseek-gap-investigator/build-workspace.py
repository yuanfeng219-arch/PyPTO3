"""Static, non-executing source/IR index. Never imports the inspected Python.

Generated JSON is derived data, separate from immutable Data/DeepseekV4 inputs.
Edges encode lexical value use, control containment, and visible statement order.
They do not model runtime scheduling, loop-carried dependencies, or memory effects.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent / 'Data' / 'DeepseekV4'
TARGETS = {'a2a': 'o_group_a2a_wait', 'token': 'cp_token_allgather_payload_wait'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def extract(text, path, target):
    tree = ast.parse(text)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    container = next((n for n in functions if n.name == target), None)
    function = container.name if container else None
    if container is None:
        for fn in functions:
            container = next((n for n in ast.walk(fn) if isinstance(n, ast.With)
                              and any(isinstance(c, ast.Constant) and c.value == target
                                      for item in n.items for c in ast.walk(item.context_expr))), None)
            if container:
                function = fn.name
                break
    if container is None:
        return {'nodes': [], 'edges': [], 'warnings': ['该快照未找到精确函数或 scope 名称；未做模糊映射。'], 'file': path}
    nodes, edges, warnings = [], [], ['静态局部图；连线表示数据引用、控制嵌套或源码顺序，不代表运行时调度与耗时因果。']
    env = {}

    def node(n, label, kind='op', symbol=None):
        ident = f'{function}:{n.lineno}:{len(nodes)}'
        nodes.append({'id': ident, 'type': kind, 'label': label, 'symbol': symbol,
                      'source': {'file': path, 'line': n.lineno, 'end_line': n.end_lineno,
                                 'function': function, 'scope': target},
                      'text': ast.get_source_segment(text, n), 'signature': ast.dump(n, include_attributes=False)})
        return ident

    def refs(expr, target_id, scope_env, edge_kind='data'):
        if expr is None:
            return
        for item in ast.walk(expr):
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id not in {'pl', 'pld', 'self'}:
                if item.id not in scope_env:
                    scope_env[item.id] = node(item, item.id, 'incast', item.id)
                edge = {'source': scope_env[item.id], 'target': target_id, 'kind': edge_kind,
                        'symbol': item.id, 'source_ref': {'file': path, 'line': item.lineno}}
                if edge not in edges:
                    edges.append(edge)

    def walk(body, scope_env, parent=None):
        for stmt in body:
            if isinstance(stmt, (ast.Return, ast.Pass)):
                continue
            if isinstance(stmt, (ast.For, ast.If, ast.With)):
                expr = stmt.iter if isinstance(stmt, ast.For) else stmt.test if isinstance(stmt, ast.If) else stmt.items[0].context_expr
                ident = node(stmt, type(stmt).__name__ + ' · ' + ast.unparse(expr), 'op')
                refs(expr, ident, scope_env)
                if parent:
                    edges.append({'source': parent, 'target': ident, 'kind': 'control', 'source_ref': {'file': path, 'line': stmt.lineno}})
                child = scope_env.copy()
                if isinstance(stmt, ast.For):
                    for name in ast.walk(stmt.target):
                        if isinstance(name, ast.Name):
                            child[name.id] = ident
                walk(stmt.body, child, ident)
                if getattr(stmt, 'orelse', None):
                    walk(stmt.orelse, scope_env.copy(), ident)
                # Assignments from conditional/loop bodies are intentionally not propagated.
                continue
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Expr)):
                warnings.append(f'未支持 {type(stmt).__name__} · L{stmt.lineno}')
                continue
            value = stmt.value
            if value is None:
                continue
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
            symbols = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
            label = ast.unparse(value.func) if isinstance(value, ast.Call) else type(value).__name__
            ident = node(stmt, label, 'op', ', '.join(symbols) or None)
            if isinstance(value, ast.Call):
                for arg in value.args:
                    refs(arg, ident, scope_env)
                for kw in value.keywords:
                    refs(kw.value, ident, scope_env, 'control' if kw.arg == 'deps' else 'data')
            else:
                refs(value, ident, scope_env)
            if parent:
                edges.append({'source': parent, 'target': ident, 'kind': 'control', 'source_ref': {'file': path, 'line': stmt.lineno}})
            for symbol in symbols:
                scope_env[symbol] = ident

    walk(container.body, env)

    # A side-effect-only statement can have no value input and live outside a
    # control block. Keep it connected to the preceding operation so the graph
    # does not present a false orphan. This is lexical order, not a data edge.
    incoming = {edge['target'] for edge in edges}
    operations = [item for item in nodes if item['type'] == 'op']
    for current in operations[1:]:
        if current['id'] in incoming:
            continue
        previous = [item for item in operations
                    if item['source']['line'] < current['source']['line']]
        if not previous:
            continue
        source = max(previous, key=lambda item: item['source']['line'])
        edges.append({'source': source['id'], 'target': current['id'], 'kind': 'control',
                      'symbol': '源码顺序',
                      'source_ref': {'file': path, 'line': current['source']['line']}})
    return {'nodes': nodes, 'edges': edges, 'warnings': warnings, 'file': path,
            'scope': target, 'function': function, 'line': container.lineno}


def main():
    archive = ROOT / '_jit_l3_decode_csa_20260903_010617'
    passes = []
    for file in sorted((archive / 'passes_dump').glob('*.py')):
        text = file.read_text()
        path = file.relative_to(ROOT).as_posix()
        graphs = {}
        for key, target in TARGETS.items():
            try:
                graphs[key] = extract(text, path, target)
            except SyntaxError as error:
                graphs[key] = {'nodes': [], 'edges': [], 'file': path, 'warnings': [f'静态解析失败 · L{error.lineno}: {error.msg}']}
        passes.append({'index': int(file.name[:2]), 'name': file.stem[3:], 'sha256': digest(file.read_bytes()), 'graphs': graphs})
    files = [{'path': f.relative_to(ROOT).as_posix(), 'sha256': digest(f.read_bytes()), 'bytes': f.stat().st_size}
             for f in sorted((ROOT / 'deepseek_v4_flash_dspark').rglob('*.py'))]
    manifest = {'schema_version': 1, 'root': 'Data/DeepseekV4', 'mode': 'source-archive', 'files': files,
                'revision': 'sha256:' + digest(json.dumps(files, sort_keys=True).encode()),
                'entry': 'deepseek_v4_flash_dspark/decode_csa.py', 'passes': len(passes),
                'traces': [f.relative_to(ROOT).as_posix() for f in sorted(archive.rglob('merged_swimlane_*.json'))],
                'execution': 'unbound', 'warnings': ['源码摘要仅标识归档，不证明与基线构建 revision 一致。', '未连接 Ascend 执行服务；golden、PyPTO 版本、输入和设备尚待验证。']}
    for filename, payload in [('pass-graphs.json', {'schema_version': 1, 'passes': passes}), ('project-manifest.json', manifest)]:
        (HERE / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passes': len(passes), 'source_files': len(files), 'traces': len(manifest['traces']),
                      'graphs': sum(bool(g['nodes']) for p in passes for g in p['graphs'].values())}))


if __name__ == '__main__':
    main()
