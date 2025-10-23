import re
import sympy as sp

# --- Normalizador de expresiones ---
def normalizar_expr(expr_str: str) -> str:
    # π -> pi
    expr_str = expr_str.replace("π", "pi")
    # ^ -> **
    expr_str = expr_str.replace("^", "**")
    # e^(...) -> exp(...)
    expr_str = re.sub(r"e\^\(([^)]+)\)", r"exp(\1)", expr_str)
    expr_str = re.sub(r"e\^([a-zA-Z0-9]+)", r"exp(\1)", expr_str)
    # ln -> log
    expr_str = expr_str.replace("ln", "log")
    # e aislada -> E (constante de Euler)
    expr_str = re.sub(r"\be\b", "E", expr_str)
    return expr_str

# --- Función con L'Hopital ---
def crear_funcion_con_lhopital(expr_str: str):
    expr_str = normalizar_expr(expr_str)
    x = sp.symbols('x')
    expr = sp.sympify(expr_str)

    def f_safe(val):
        try:
            res = expr.subs(x, val)
            if res.has(sp.zoo) or res.has(sp.nan) or res.has(sp.oo):
                res = sp.limit(expr, x, val)
            if res.is_real:
                return float(res)
            else:
                return float(sp.re(res))
        except Exception:
            return float(sp.limit(expr, x, val))
    return f_safe
