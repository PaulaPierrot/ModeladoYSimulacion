from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
import sys
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from PyQt6.QtGui import QFont

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from func_utils import normalizar_expr, crear_funcion_con_lhopital
from math_keyboard import MathKeyboard

from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor, function_exponentiation
)

try:
    from sympy.parsing.sympy_parser import implicit_application
    _HAS_IMPLICIT_APP = True
except Exception:
    _HAS_IMPLICIT_APP = False

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  
    convert_xor,                          
    function_exponentiation              
) + ((implicit_application,) if _HAS_IMPLICIT_APP else ())

_ALIASES = {
    'ln': 'log',      
    'sen': 'sin',
    'tg': 'tan',
    'ctg': 'cot',
    '√': 'sqrt',
}

def _normalize_tokens(s: str) -> str:
    s = (s or '').strip()
    s = s.replace('π', 'pi')
    for k, v in _ALIASES.items():
        s = s.replace(k, v)
    return s

def _parse_user_expr(s: str, sym_name: str = 'x'):
    """parsea aceptando multiplicación implicita, ^ y e como constante.
    Devuelve (x_symbol, expr_sympy, f_numeric)."""
    s = _normalize_tokens(s)
    x = sp.symbols(sym_name, real=True)
    ns = {
        'x': x,
        'e': sp.E, 'E': sp.E, 'pi': sp.pi,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'cot': sp.cot,
        'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
    }
    expr = parse_expr(s, local_dict=ns, transformations=_TRANSFORMS, evaluate=True)
    f_num = sp.lambdify(x, expr, 'math')
    return x, expr, f_num

def _normalize_user_expr_to_str(s: str) -> str:
    """normaliza a una cadena segura para pasar a helpers (L'Hôpital, etc.)."""
    try:
        _, expr, _ = _parse_user_expr(s)
        return str(sp.simplify(expr))
    except Exception:
        return s
    
def style_primary_button(btn, *, radius=12, accent="#0ea5e9"):
    btn.setFixedHeight(44)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: #ffffff;
            color: #0f172a;
            border: 1px solid rgba(0,0,0,0.12);
            border-radius: {radius}px;
            padding: 10px 16px;
            font-weight: 800;
        }}
        QPushButton:hover {{
            background: #f6f8fb;
            border-color: {accent};
        }}
        QPushButton:pressed {{
            background: #eef2f7;
            padding-top: 11px;        
        }}
        QPushButton:disabled {{
            color: rgba(15,23,42,0.35);
            border-color: rgba(0,0,0,0.08);
            background: #fafafa;
        }}
    """)
    shadow = QGraphicsDropShadowEffect(btn)
    shadow.setBlurRadius(22)   
    shadow.setOffset(0, 2)   
    shadow.setColor(QColor(0, 0, 0, 70))
    btn.setGraphicsEffect(shadow)    

# =============== utilidades de despeje

def _build_functions(f_str):
    return _parse_user_expr(f_str)

def _split_linear_term(f, x):
    A = sp.Wild('A', exclude=[x])
    ax = 0
    rest = f
    for t in sp.Add.make_args(sp.expand(f)):
        m = t.match(A*x)
        if m is not None and x not in m[A].free_symbols:
            ax += m[A]
            rest -= t
    return sp.simplify(ax), sp.simplify(rest)

def _find_power_terms(f, x):
    terms = []
    F = sp.expand(f)
    for t in sp.Add.make_args(F):
        if not t.has(x):
            continue
        if t.is_Pow and t.base == x and t.exp.is_integer and t.exp >= 2:
            terms.append((sp.Integer(1), int(t.exp), t))
            continue
        if t.is_Mul:
            coef = sp.Integer(1)
            power = 0
            ok = True
            for fac in t.args:
                if fac == x:
                    power += 1
                elif fac.is_Pow and fac.base == x and fac.exp.is_integer and fac.exp > 0:
                    power += int(fac.exp)
                elif fac.has(x):
                    ok = False
                    break
                else:
                    coef *= fac
            if ok and power >= 2 and x not in coef.free_symbols:
                terms.append((sp.simplify(coef), int(power), t))
    terms.sort(key=lambda z: -z[1])
    return terms

def _sup_gprime(gprime_num, a, b, grid_pts=101):
    xs = np.linspace(float(a), float(b), grid_pts)
    vals = []
    for xv in xs:
        try:
            vals.append(abs(gprime_num(xv)))
        except Exception:
            pass
    return max(vals) if vals else float('inf')

def _const_candidates_from_arithmetic_progression(base, step, intervalo, x0, label):
    out = []
    if intervalo is not None:
        a, b = intervalo
        kmin = int(sp.ceiling((a - base)/step))
        kmax = int(sp.floor((b - base)/step))
        for k in range(kmin, kmax+1):
            const = sp.simplify(base + k*step)
            out.append((f"{label}; k={k}", sp.simplify(const)))
    else:
        k_star = int(round(float(sp.N((x0 - base)/step))))
        const = sp.simplify(base + k_star*step)
        out.append((f"{label}; k≈{k_star}", sp.simplify(const)))
    return out

def _extract_single_trig_zero_candidates(f, x, intervalo, x0):
    cand = []
    A = sp.factor(f)
    if A.func == sp.cos or (A.is_Mul and any(arg.func == sp.cos for arg in A.args)):
        arg = A.args[0] if A.func == sp.cos else next(arg.args[0] for arg in A.args if arg.func == sp.cos)
        B = sp.diff(arg, x)
        if B.free_symbols == set() and B != 0:
            C = sp.simplify(arg - B*x)
            base = sp.simplify((sp.pi/2 - C)/B)
            paso = sp.simplify(sp.pi/B)
            cand += _const_candidates_from_arithmetic_progression(
                base, paso, intervalo, x0, "Despeje cos: x = pi/2 + k*pi (ajustada por B,C)"
            )
    if A.func == sp.sin or (A.is_Mul and any(arg.func == sp.sin for arg in A.args)):
        arg = A.args[0] if A.func == sp.sin else next(arg.args[0] for arg in A.args if arg.func == sp.sin)
        B = sp.diff(arg, x)
        if B.free_symbols == set() and B != 0:
            C = sp.simplify(arg - B*x)
            base = sp.simplify((-C)/B)
            paso = sp.simplify(sp.pi/B)
            cand += _const_candidates_from_arithmetic_progression(
                base, paso, intervalo, x0, "Despeje sin: x = k*pi (ajustada por B,C)"
            )
    return cand

# ==================== 

def punto_fijo_unificado(
    x0,
    tol=1e-6,
    maxiter=200,
    g_str=None,
    f_str=None,
    intervalo=None,
    grid_pts=101,
):
    """Devuelve: raiz, historia(list[dict]), g_plot (callable o None), g_expr (sympy o None), modo(str)"""
    x0_val = float(x0)

    # --- modo g(x) directa ---
    if g_str and g_str.strip():
        g = crear_funcion_con_lhopital(_normalize_user_expr_to_str(g_str))
        x = sp.symbols('x')
        try:
            g_sym = sp.sympify(_normalize_user_expr_to_str(g_str), locals={'x': x, 'pi': sp.pi, 'E': sp.E})
        except Exception:
            g_sym = None
        f_disp = (g_sym - x) if g_sym is not None else None
        f_disp_num = (sp.lambdify(x, f_disp, 'math') if f_disp is not None else None)

        historia = []
        xk = x0_val
        for k in range(1, maxiter + 1):
            xnext = float(g(xk))
            err = abs(xnext - xk)
            fval = (float(f_disp_num(xnext)) if f_disp_num else None)
            historia.append({'k': k, 'xk': xk, 'xnext': xnext, 'err': err, 'fval': fval})
            if err < tol:
                return xnext, historia, g, g_sym, 'g_directa'
            xk = xnext
        return xk, historia, g, g_sym, 'g_directa'

    # --- modo f(x) con despeje ---
    if not (f_str and f_str.strip()):
        raise ValueError("Debes ingresar g(x) o f(x).")

    x, f, f_num = _build_functions(f_str)

    candidatos = []
    candidatos += _extract_single_trig_zero_candidates(f, x, intervalo, x0_val)

    a_lin, R = _split_linear_term(f, x)
    if a_lin != 0:
        g1 = sp.simplify(-R / a_lin)
        candidatos.append(("Despeje lineal: a*x + R = 0 → x = -R/a", g1))

    for a_pow, n_pow, term in _find_power_terms(f, x):
        Rpow = sp.simplify(f - term)
        base = sp.simplify(-Rpow / a_pow)
        gpos = sp.simplify(sp.Pow(base, sp.Rational(1, n_pow)))
        candidatos.append((f"Despeje potencia n={n_pow}: x = ((-R)/a)^(1/{n_pow})", gpos))
        if n_pow % 2 == 0:
            candidatos.append((f"Despeje potencia n={n_pow} (rama -)", -gpos))

    if not candidatos:
        raise ValueError("No se pudo generar ningún despeje g(x) a partir de f(x). Probá con Newton.")

    elegido = None
    for desc, gexpr in candidatos:
        try:
            gp = sp.diff(gexpr, x)
            g_num  = sp.lambdify(x, gexpr, 'math')
            gp_num = sp.lambdify(x, gp,     'math')
            if intervalo is None:
                q = abs(float(gp_num(x0_val)))
                ok = q < 1
            else:
                a, b = intervalo
                q = _sup_gprime(gp_num, a, b, grid_pts)
                ok = q < 1
        except Exception:
            ok = False
        if ok:
            elegido = (desc, gexpr, g_num)
            break

    if elegido is None:
        raise ValueError("Ningún despeje g(x) cumple |g'|<1 cerca de x0. Probá con otro x0 o método.")

    desc, gexpr, g_num = elegido

    historia = []
    xk = x0_val
    for k in range(1, maxiter + 1):
        try:
            xnext = float(g_num(xk))
        except Exception:
            raise ValueError("La evaluación numérica de g(x) falló durante la iteración.")
        err = abs(xnext - xk)
        fval = float(f_num(xnext))
        historia.append({'k': k, 'xk': xk, 'xnext': xnext, 'err': err, 'fval': fval})
        if err < tol:
            try:
                g_plot = crear_funcion_con_lhopital(str(gexpr))
            except Exception:
                g_plot = lambda t: float(sp.N(g_num(t)))
            return xnext, historia, g_plot, sp.simplify(gexpr), f"despeje: {desc}"
        xk = xnext

    try:
        g_plot = crear_funcion_con_lhopital(str(gexpr))
    except Exception:
        g_plot = lambda t: float(sp.N(g_num(t)))
    return xk, historia, g_plot, sp.simplify(gexpr), f"despeje: {desc}"

# =============================== UI 

class PuntoFijoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Punto Fijo – g(x) directa o despeje desde f(x)")
        self.resize(1000, 720)
        self.active_input = None

        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)

        row = QHBoxLayout(); root.addLayout(row)
        self.g_input  = QLineEdit("")  # vacio => usa f(x) con despeje
        self.f_input  = QLineEdit("2e^x^2-5x")
        self.p0_input = QLineEdit("0")
        self.tol_input= QLineEdit("0.0001")
        for e in [self.g_input, self.f_input, self.p0_input, self.tol_input]:
            e.installEventFilter(self)
        row.addWidget(QLabel("g(x):")); row.addWidget(self.g_input)
        row.addWidget(QLabel("f(x):")); row.addWidget(self.f_input)
        row.addWidget(QLabel("x₀:"));   row.addWidget(self.p0_input)
        row.addWidget(QLabel("Tolerancia:")); row.addWidget(self.tol_input)

        hint = QLabel("Si g(x) tiene texto → usa g directa. Si g(x) está vacío → intenta despejar desde f(x).")
        root.addWidget(hint)

        self.keyboard = MathKeyboard(self.insertar_texto)
        root.addWidget(self.keyboard)

        self.btn = QPushButton("Calcular raíz (Punto Fijo)")
        style_primary_button(self.btn) 
        self.btn.clicked.connect(self.calcular)
        root.addWidget(self.btn)

        self.figure, self.ax = plt.subplots(figsize=(8,5))
        self.canvas = FigureCanvas(self.figure)
        root.addWidget(self.canvas)

        self.output = QTextEdit(); self.output.setReadOnly(True)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self.output.setFont(font)
        self.output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        root.addWidget(self.output)

    def eventFilter(self, obj, event):
        if event.type() == event.Type.FocusIn:
            self.active_input = obj
        return super().eventFilter(obj, event)

    def insertar_texto(self, texto):
        if not self.active_input:
            return
        cur = self.active_input.cursorPosition()
        s = self.active_input.text()
        self.active_input.setText(s[:cur] + texto + s[cur:])
        self.active_input.setCursorPosition(cur + len(texto))

    def calcular(self):
        try:
            g_str = self.g_input.text().strip()
            f_str = self.f_input.text().strip()

            x0 = float(sp.sympify(normalizar_expr(self.p0_input.text()), locals={'pi': sp.pi, 'E': sp.E}))
            tol = float(sp.sympify(normalizar_expr(self.tol_input.text()), locals={'pi': sp.pi, 'E': sp.E}))

            raiz, historia, g_plot, g_expr, modo = punto_fijo_unificado(
                x0=x0, tol=tol, g_str=g_str if g_str else None, f_str=f_str if f_str else None
            )

            f_plot = None
            if f_str:
                try:
                    f_plot = crear_funcion_con_lhopital(_normalize_user_expr_to_str(f_str))
                except Exception:
                    f_plot = None

            center = raiz if np.isfinite(raiz) else x0
            X = np.linspace(center - 2, center + 2, 500)

            self.ax.clear()
            self.ax.plot(X, X, 'r--', label='y = x')
            if g_plot is not None:
                Yg = [g_plot(t) for t in X]
                self.ax.plot(X, Yg, 'b', label='g(x)')
            if f_plot is not None:
                Yf = [f_plot(t) for t in X]
                self.ax.plot(X, Yf, 'm', label='f(x)')
                self.ax.axhline(0, color='k', linewidth=0.8)

            try:
                self.output.append(f"Raíz aproximada = {raiz:.12f}")
                self.output.append("")
                self.output.append("Iteraciones:")
                for st in historia:
                    k, xk, xn, err = st['k'], st['xk'], st['xnext'], st['err']
                    fval = st.get('fval', None)
                    # 👇 ahora mostramos g(x) también
                    line = f"k={k:2d}:  x = {xk: .12f},  g(x) = {xn: .12f},  |Δ| = {err: .3e}"
                    if fval is not None:
                        line += f",  f(x) = {fval: .3e}"
                    self.output.append(line)
                # Al final, evaluamos g en la raíz aproximada
                '''if g_expr is not None:
                    try:
                        x = sp.symbols('x')
                        g_eval = sp.lambdify(x, g_expr, 'math')
                        g_final = g_eval(raiz)
                        self.output.append("")
                        self.output.append(f"En la última iteración: g({raiz:.12f}) = {g_final:.12f}")
                    except Exception:
                        pass'''


            except Exception:
                pass

            self.ax.axvline(raiz, color='g', linestyle='--', label=f"Raíz ≈ {raiz:.8f}")
            title = "Punto Fijo (" + ("g directa" if modo == 'g_directa' else str(modo)) + ")"
            if g_expr is not None:
                title += f"\nUsando g(x) = {sp.simplify(g_expr)}"
            self.ax.set_title(title)
            self.ax.grid(True)
            self.ax.legend()
            self.canvas.draw()

            
            '''self.output.clear()
            self.output.append(f"Raíz aproximada = {raiz:.12f}")
            self.output.append("")
            self.output.append("Iteraciones:")
            for st in historia:
                k, xk, xn, err = st['k'], st['xk'], st['xnext'], st['err']
                fval = st.get('fval', None)
                line = f"k={k:2d}:  x = {xk: .12f},  |Δ| = {err: .3e}"
                if fval is not None:
                    line += f",  f(x) = {fval: .3e}"
                self.output.append(line)'''

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = PuntoFijoWindow()
    w.show()
    sys.exit(app.exec())
