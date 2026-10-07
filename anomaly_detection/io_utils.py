import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class _WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
                if not node.body and not isinstance(node, ast.Module):
                    node.body.append(ast.Pass())
        return node


def source_identity(source):
    tree = _WithoutDocstrings().visit(ast.parse(source))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def code_identity(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    paths = list(root.glob("*.py")) + list((root / "structured_privacy").glob("*.py"))
    return "ast-v1:" + fingerprint({
        str(p.relative_to(root)): source_identity(p.read_text(encoding="utf-8"))
        for p in sorted(paths)
    })


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def epsilon_tag(value):
    return format(value, ".12g")
