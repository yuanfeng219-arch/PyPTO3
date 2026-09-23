"""Synthetic extraction tests. No supplied Python artifact is executed."""
import ast
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("extract", Path(__file__).with_name("analyze-stage2-cfg.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def graph(source):
    return module.CFG(source, ast.parse(source).body[0], 1, []).result()


class NormalFlowTests(unittest.TestCase):
    def test_break_continue_and_for_else(self):
        g = graph("def f(n):\n    for i in range(n):\n        if i > 0:\n            continue\n        else:\n            break\n    else:\n        x = 1\n    return n\n")
        node = lambda kind: next(n for n in g["nodes"] if n["kind"] == kind)
        successor = lambda n: next(e["target"] for e in g["edges"] if e["source"] == n["id"])
        self.assertEqual(successor(node("continue")), node("loop")["id"])
        self.assertEqual(successor(node("break")), node("return")["id"])
        exhausted = next(e for e in g["edges"] if e["source"] == node("loop")["id"] and e["label"] == "迭代结束")
        self.assertEqual(exhausted["target"], node("statements")["id"])

    def test_unhandled_compound_is_not_silently_dropped(self):
        with self.assertRaises(module.Unsupported):
            graph("def f():\n    try:\n        x = 1\n    except Exception:\n        return 2\n")

    def test_submit_is_separate_from_successor(self):
        g = graph("def f():\n    x = 1\n    pl.submit(self.kernel, x)\n    return x\n")
        submit = next(n for n in g["nodes"] if n["kind"] == "submit")
        self.assertEqual(submit["submitTarget"], "kernel")
        edge = next(e for e in g["edges"] if e["source"] == submit["id"])
        successor = next(n for n in g["nodes"] if n["id"] == edge["target"])
        self.assertEqual(successor["kind"], "return")


if __name__ == "__main__":
    unittest.main()
