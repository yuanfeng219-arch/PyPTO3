"""Extract lexical normal-flow graphs from Python AST without executing artifacts.

The frontend snapshot and five snapshots needed by the three reviewed cases are analyzed.
These are statement/region graphs, NOT backend LLVM basic blocks. Exceptional
flow and the internal behavior of PyPTO operations/context managers are excluded.
Unsupported compound statements fail the selected function, never disappear.
Output is consumed as data and embedded in the canonical standalone extension.
"""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys


class Unsupported(Exception):
    pass


class CFG:
    def __init__(self, source, fn, snapshot, entities):
        self.source = source.splitlines()
        self.fn = fn
        self.snapshot = snapshot
        self.entities = entities
        self.nodes = []
        self.edges = []
        self.exit = self.add("exit", fn.end_lineno, fn.end_lineno, "EXIT")
        first = self.sequence(fn.body, self.exit)
        self.entry = self.add("entry", fn.lineno, fn.lineno, "ENTRY · " + fn.name)
        self.edge(self.entry, first)

    def add(self, kind, start, end, text=None, **extra):
        ident = len(self.nodes)
        region = [e["id"] for e in self.entities if e["snapshot"] == self.snapshot
                  and e["start"] <= start <= e["end"] and end <= e["end"]]
        if kind == "exit":
            region = []
        if kind == "return":
            # The outline proof explicitly excludes the appended return.
            region = [ident for ident in region if not any(
                e["id"] == ident and e["kind"] == "function" and end == e["end"]
                for e in self.entities)]
        self.nodes.append(dict(id=ident, kind=kind, start=start, end=end,
                               text=text or "\n".join(self.source[start-1:end]),
                               entities=region, **extra))
        return ident

    def edge(self, source, target, label=""):
        self.edges.append(dict(id=len(self.edges), source=source, target=target, label=label))

    def sequence(self, statements, following, loop=None):
        # Reverse construction preserves explicit successors, including early exits.
        pending = []

        def region(stmt):
            # Byte-identical 02/03 must use the same block partition even though
            # 03 has no separate semantic entity records.
            partition_snapshot = 2 if self.snapshot == 3 else self.snapshot
            return tuple(e["id"] for e in self.entities if e["snapshot"] == partition_snapshot
                         and e["start"] <= stmt.lineno and stmt.end_lineno <= e["end"])

        def flush(target):
            if not pending:
                return target
            block = list(reversed(pending))
            node = self.add("statements", block[0].lineno, block[-1].end_lineno,
                            statementCount=len(block))
            self.edge(node, target)
            pending.clear()
            return node

        for stmt in reversed(statements):
            simple = isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Pass))
            submit = any(isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.submit" for n in ast.walk(stmt))
            if simple and not submit:
                if pending and (len(pending) == 8 or region(stmt) != region(pending[-1])):
                    following = flush(following)
                pending.append(stmt)
                continue
            following = flush(following)
            following = self.statement(stmt, following, loop)
        return flush(following)

    def statement(self, stmt, following, loop):
        if isinstance(stmt, (ast.For, ast.While)):
            header = self.add("loop", stmt.lineno, stmt.lineno,
                              self.source[stmt.lineno-1].strip())
            exhausted = self.sequence(stmt.orelse, following, loop)
            body = self.sequence(stmt.body, header, (header, following))
            self.edge(header, body, "有下一项" if isinstance(stmt, ast.For) else "True")
            self.edge(header, exhausted, "迭代结束" if isinstance(stmt, ast.For) else "False")
            return header
        if isinstance(stmt, ast.If):
            header = self.add("branch", stmt.lineno, stmt.lineno, ast.unparse(stmt.test))
            self.edge(header, self.sequence(stmt.body, following, loop), "True")
            self.edge(header, self.sequence(stmt.orelse, following, loop), "False")
            return header
        if isinstance(stmt, ast.With):
            end = self.add("scope_exit", stmt.end_lineno, stmt.end_lineno, "离开词法 scope（正常流）")
            self.edge(end, following)
            start = self.add("scope", stmt.lineno, stmt.lineno,
                             "with " + ", ".join(ast.unparse(i) for i in stmt.items))
            self.edge(start, self.sequence(stmt.body, end, loop))
            return start
        if isinstance(stmt, (ast.Return, ast.Break, ast.Continue)):
            if not isinstance(stmt, ast.Return) and loop is None:
                raise Unsupported("loop control without enclosing loop")
            node = self.add(type(stmt).__name__.lower(), stmt.lineno, stmt.end_lineno)
            target = self.exit if isinstance(stmt, ast.Return) else loop[1 if isinstance(stmt, ast.Break) else 0]
            self.edge(node, target, "return" if isinstance(stmt, ast.Return) else type(stmt).__name__.lower())
            return node
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Pass)):
            raise Unsupported(f"{type(stmt).__name__} at L{stmt.lineno}")
        # Calls are operations within the lexical CFG, never interprocedural CFG edges.
        calls = [n for n in ast.walk(stmt) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == "pl.submit"]
        target = ast.unparse(calls[0].args[0]).removeprefix("self.") if len(calls) == 1 and calls[0].args else None
        node = self.add("submit" if target else "statement", stmt.lineno, stmt.end_lineno,
                        submitTarget=target)
        self.edge(node, following)
        return node

    def result(self):
        ids = {n["id"] for n in self.nodes}
        assert all(e["source"] in ids and e["target"] in ids for e in self.edges)
        assert len([n for n in self.nodes if n["kind"] == "entry"]) == 1
        assert not any(e["source"] == self.exit for e in self.edges)
        return dict(status="analyzed", nodes=self.nodes, edges=self.edges,
                    entry=self.entry, exit=self.exit)


def extract(root, evidence):
    graphs = {}
    for index in (0, 1, 2, 3, 8, 9):
        snapshot = evidence["snapshots"][index]
        raw = (root / snapshot["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == snapshot["sha256"]
        source = raw.decode()
        functions = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)]
        graphs[str(index)] = {}
        for fn in functions:
            try:
                graph = CFG(source, fn, index, evidence["entities"]).result()
            except Unsupported as exc:
                graph = dict(status="unsupported", reason=str(exc))
            graphs[str(index)][fn.name] = dict(name=fn.name, start=fn.lineno, end=fn.end_lineno, **graph)
    return dict(schema="stage2_lexical_cfg.v1", granularity="AST statements and lexical regions",
                scope="normal control flow only; no exception, context-manager internals or backend scheduling semantics",
                graphs=graphs)


if __name__ == "__main__":
    evidence = json.loads(Path(sys.argv[2]).read_text())
    data = extract(Path(sys.argv[1]), evidence)
    print(base64.b64encode(gzip.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0)).decode())
