import ast
import builtins
import time
from typing import Any, Dict

class Executor:
    ALLOWED_BUILTINS = {
        "range", "len", "min", "max", "sum", "sorted", "enumerate", "any", "all",
        "float", "int", "str", "list", "dict", "set", "tuple", "abs", "zip", "print"
    }

    ALLOWED_AST_NODES = (
        ast.Module, ast.Assign, ast.AugAssign, ast.AnnAssign,
        ast.Expr, ast.Call, ast.Load, ast.Store,
        ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        ast.If, ast.For, ast.While, ast.Break, ast.Continue,
        ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice,
        ast.Attribute, ast.Name, ast.Constant, ast.Return,
        ast.comprehension, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        ast.With, ast.IfExp, ast.Pass, ast.JoinedStr, ast.FormattedValue,
        # Comparison operators
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
        # Binary operators
        ast.Add, ast.Sub, ast.Mult, ast.MatMult, ast.Div, ast.Mod, ast.Pow, ast.LShift, ast.RShift,
        ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv,
        # Unary operators
        ast.Invert, ast.Not, ast.UAdd, ast.USub,
        # Boolean operators
        ast.And, ast.Or,
    )

    def __init__(self, time_limit_s: float = 5.0):
        self.time_limit_s = time_limit_s

    def _validate_ast(self, tree: ast.AST):
        for node in ast.walk(tree):
            if not isinstance(node, self.ALLOWED_AST_NODES):
                raise RuntimeError(f"Disallowed AST node: {type(node).__name__}")

    def run(self, program: str, env: Dict[str, Any]) -> Any:
        # Parse & validate AST
        tree = ast.parse(program, mode="exec")
        self._validate_ast(tree)

        # Build restricted globals
        safe_builtins = {k: getattr(builtins, k) for k in self.ALLOWED_BUILTINS}
        globals_env: Dict[str, Any] = {"__builtins__": safe_builtins}
        globals_env.update(env)

        # Time limit via watchdog
        start = time.time()
        def _check_time():
            if time.time() - start > self.time_limit_s:
                raise TimeoutError("Program exceeded time limit")

        # Inject a simple ticker the plan can call if desired
        globals_env["_tick"] = _check_time

        # Execute
        locals_env: Dict[str, Any] = {}
        exec(compile(tree, filename="<plan>", mode="exec"), globals_env, locals_env)
        _check_time()
        return