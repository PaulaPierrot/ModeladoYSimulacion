from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QCheckBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import sympy as sp

# ── tus utilidades (l'Hôpital y normalizador) ─────────────────────────
from func_utils import normalizar_expr, crear_funcion_con_lhopital
# from math_keyboard import MathKeyboard   # opcional

matplotlib.use("QtAgg")

# ================ helpers UI/estilo ================
NEAR_ZERO_TOL = 1e-12

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

# ================ helpers matemáticos ================
def fmt(v, nd=10):
    try:
        vv = float(v)
    except Exception:
        vv = float(sp.N(v))
    if abs(vv) < NEAR_ZERO_TOL: vv = 0.0
    return f"{vv:.{nd}g}"

def mt_wrap(tex_line: str) -> str:
    if not tex_line: return ""
    s = tex_line.replace(r"\displaystyle", "").strip()
    if s.startswith("$") and s.endswith("$"): return s
    return f"${s}$"

def mt_block(lines) -> str:
    return "\n".join(mt_wrap(L) for L in lines)


# ---- Diferencias finitas (1ª, 2ª y 3ª) con “paso a paso” ----
def fd_first(f, x0, h, o2_forward_backward=False):
    f0  = f(x0); fph = f(x0 + h); fmh = f(x0 - h)
    pasos = []
    if o2_forward_backward:
        f2ph = f(x0 + 2*h); f2mh = f(x0 - 2*h)
        prog = (-3*f0 + 4*fph - f2ph) / (2*h)   # O(h^2)
        regr = ( 3*f0 - 4*fmh + f2mh) / (2*h)   # O(h^2)
        pasos.append(f"Progresiva (O(h²)): (-3f(x₀)+4f(x₀+h)-f(x₀+2h))/(2h) = (-3·{fmt(f0)} + 4·{fmt(fph)} - {fmt(f2ph)}) / (2·{fmt(h)}) = {fmt(prog)}")
        pasos.append(f"Regresiva  (O(h²)): (3f(x₀)-4f(x₀-h)+f(x₀-2h))/(2h)  = ( 3·{fmt(f0)} - 4·{fmt(fmh)} + {fmt(f2mh)}) / (2·{fmt(h)}) = {fmt(regr)}")
    else:
        prog = (fph - f0) / h                   # O(h)
        regr = (f0 - fmh) / h                   # O(h)
        pasos.append(f"Progresiva (O(h)):  (f(x₀+h)-f(x₀))/h = ({fmt(fph)} - {fmt(f0)}) / {fmt(h)} = {fmt(prog)}")
        pasos.append(f"Regresiva  (O(h)):  (f(x₀)-f(x₀-h))/h = ({fmt(f0)} - {fmt(fmh)}) / {fmt(h)} = {fmt(regr)}")
    cent = (fph - fmh) / (2*h)                  # O(h^2)
    pasos.append(f"Central    (O(h²)): (f(x₀+h)-f(x₀-h))/(2h) = ({fmt(fph)} - {fmt(fmh)}) / (2·{fmt(h)}) = {fmt(cent)}")
    return {"prog": prog, "regr": regr, "cent": cent}, pasos

def fd_second(f, x0, h):
    f0=f(x0); fph=f(x0+h); fmh=f(x0-h); f2ph=f(x0+2*h); f2mh=f(x0-2*h)
    prog = (f2ph - 2*fph + f0)/(h*h)
    regr = (f0 - 2*fmh + f2mh)/(h*h)
    cent = (fph - 2*f0 + fmh)/(h*h)
    pasos = [
        f"Progresiva: (f(x₀+2h)-2 f(x₀+h)+f(x₀))/h²  = ({fmt(f2ph)} - 2·{fmt(fph)} + {fmt(f0)}) / {fmt(h)}² = {fmt(prog)}",
        f"Regresiva:  (f(x₀)-2 f(x₀-h)+f(x₀-2h))/h²  = ({fmt(f0)} - 2·{fmt(fmh)} + {fmt(f2mh)}) / {fmt(h)}² = {fmt(regr)}",
        f"Central:    (f(x₀+h)-2 f(x₀)+f(x₀-h))/h²   = ({fmt(fph)} - 2·{fmt(f0)} + {fmt(fmh)}) / {fmt(h)}² = {fmt(cent)}",
    ]
    return {"prog": float(prog), "regr": float(regr), "cent": float(cent)}, pasos

def fd_third(f, x0, h):
    f0=f(x0); f1=f(x0+h); f2=f(x0+2*h); f3=f(x0+3*h)
    b1=f(x0-h); b2=f(x0-2*h); b3=f(x0-3*h)
    prog = (f3 - 3*f2 + 3*f1 - f0)/(h**3)
    regr = (f0 - 3*b1 + 3*b2 - b3)/(h**3)
    cent = (b2 - 2*b1 + 2*f1 - f2)/(2*h**3)
    pasos = [
        f"Progresiva: (f(x₀+3h)-3f(x₀+2h)+3f(x₀+h)-f(x₀))/h³   = ({fmt(f3)} - 3·{fmt(f2)} + 3·{fmt(f1)} - {fmt(f0)}) / {fmt(h)}³ = {fmt(prog)}",
        f"Regresiva:  (f(x₀)-3f(x₀-h)+3f(x₀-2h)-f(x₀-3h))/h³    = ({fmt(f0)} - 3·{fmt(b1)} + 3·{fmt(b2)} - {fmt(b3)}) / {fmt(h)}³ = {fmt(regr)}",
        f"Central:    (f(x₀-2h)-2f(x₀-h)+2f(x₀+h)-f(x₀+2h))/(2h³) = ({fmt(b2)} - 2·{fmt(b1)} + 2·{fmt(f1)} - {fmt(f2)}) / (2·{fmt(h)}³) = {fmt(cent)}",
    ]
    return {"prog": float(prog), "regr": float(regr), "cent": float(cent)}, pasos

# ====== MODO TABLA para ejercicios 6 y 7 ======
def _parse_list(texto: str):
    if not texto.strip():
        return []
    partes = [p for p in texto.replace(";", ",").split(",") if p.strip()]
    vals = [float(sp.N(sp.sympify(normalizar_expr(p.strip()), locals={"pi": sp.pi, "E": sp.E}))) for p in partes]
    return vals

def _quad_local_derivs_steps(ts, xs, i):
    """
    Derivadas 1ª y 2ª en t_i ajustando un polinomio cuadrático local
    (tres puntos). Devuelve también el sistema A,b, los coeficientes y qué ventana se usó.
    """
    n = len(ts)
    if i <= 0:
        idx = [0, 1, 2]
    elif i >= n-1:
        idx = [n-3, n-2, n-1]
    else:
        idx = [i-1, i, i+1]

    t0, t1, t2 = (float(ts[k]) for k in idx)
    x0, x1, x2 = (float(xs[k]) for k in idx)

    # Sistema lineal para p(t)=a t^2 + b t + c
    A = np.array([[t0**2, t0, 1.0],
                  [t1**2, t1, 1.0],
                  [t2**2, t2, 1.0]], dtype=float)
    bvec = np.array([x0, x1, x2], dtype=float)

    # Resolver (si es mal condicionado, fallback a polyfit)
    try:
        a, bcoef, c = np.linalg.solve(A, bvec)
    except Exception:
        a, bcoef, c = np.polyfit([t0, t1, t2], [x0, x1, x2], 2)

    ti = float(ts[i])
    v = 2*a*ti + bcoef
    a2 = 2*a

    return v, a2, idx, (A, bvec), (a, bcoef, c), ti, (t0, t1, t2), (x0, x1, x2)

def _uniform_step(ts, tol=1e-9):
    """Devuelve (es_uniforme, h) para la malla de tiempos."""
    ts = np.asarray(ts, float)
    d = np.diff(ts)
    if len(d) == 0:
        return False, None
    h = float(np.median(d))
    return bool(np.all(np.abs(d - h) <= tol)), h

def _explain_step_block(i, ts, xs, vi, ai, idx, ti, method, uniform_h):
    """
    Paso a paso con despeje explícito de a,b,c usando eliminación:
      - resto ecuaciones para eliminar c
      - divido por Δt para linealizar en a y b
      - resto otra vez para despejar a
      - saco b y c
      - derivo para v(t_i)=2a t_i + b y a(t_i)=2a
      - si la malla es uniforme, comparo con fórmulas DF
    """
    # puntos de la ventana
    t0, t1, t2 = [float(ts[k]) for k in idx]
    x0, x1, x2 = [float(xs[k]) for k in idx]

    # diferencias y sumas
    D10 = t1 - t0
    D21 = t2 - t1
    S10 = t1 + t0
    S21 = t2 + t1
    X10 = x1 - x0
    X21 = x2 - x1

    # cocientes (para escribir el despeje de forma clara)
    R10 = X10 / D10
    R21 = X21 / D21

    # despeje numérico
    a = (R21 - R10) / (S21 - S10)
    b = R10 - a*S10
    c = x0 - a*(t0**2) - b*t0

    L = []
    L.append(f"{i}.  método = {method}")
    L.append(f"    Ventana usada: ({fmt(t0)},{fmt(x0)}), ({fmt(t1)},{fmt(x1)}), ({fmt(t2)},{fmt(x2)})")
    L.append("    Sistema (pasa por los 3 puntos):")
    L.append(f"       a·t0^2 + b·t0 + c = x0  →  {fmt(t0**2)}·a + {fmt(t0)}·b + c = {fmt(x0)}")
    L.append(f"       a·t1^2 + b·t1 + c = x1  →  {fmt(t1**2)}·a + {fmt(t1)}·b + c = {fmt(x1)}")
    L.append(f"       a·t2^2 + b·t2 + c = x2  →  {fmt(t2**2)}·a + {fmt(t2)}·b + c = {fmt(x2)}")
    L.append("    Restando (elimino c):")
    L.append(f"       a·(t1^2 - t0^2) + b·(t1 - t0) = x1 - x0  →  a·{fmt(t1**2 - t0**2)} + b·{fmt(D10)} = {fmt(X10)}")
    L.append(f"       a·(t2^2 - t1^2) + b·(t2 - t1) = x2 - x1  →  a·{fmt(t2**2 - t1**2)} + b·{fmt(D21)} = {fmt(X21)}")
    L.append("    Divido por Δt (para linealizar en a y b):")
    L.append(f"       a·({fmt(S10)}) + b = {fmt(R10)}")
    L.append(f"       a·({fmt(S21)}) + b = {fmt(R21)}")
    L.append("    Restando ambas:")
    L.append(f"       a·({fmt(S21 - S10)}) = {fmt(R21)} - {fmt(R10)} = {fmt(R21 - R10)}")
    L.append(f"       ⇒ a = {fmt(a)}")
    L.append(f"       ⇒ b = {fmt(R10)} - a·{fmt(S10)} = {fmt(b)}")
    L.append(f"       Con la 1ª: c = x0 - a·t0^2 - b·t0 = {fmt(c)}")
    L.append("    Derivo el cuadrático:")
    L.append(f"       v(t_i) = 2a·t_i + b = 2·{fmt(a)}·{fmt(ti)} + {fmt(b)} = {fmt(vi)}")
    L.append(f"       a(t_i) = 2a = {fmt(2*a)}")

    # comparación DF si la malla es uniforme
    is_uniform, h = uniform_h
    if is_uniform:
        if i == 0:
            v_fd = (-3*xs[0] + 4*xs[1] - xs[2])/(2*h)
            a_fd = (xs[0] - 2*xs[1] + xs[2])/(h*h)
            L.append(f"    (malla uniforme h={fmt(h)}) DF: v0≈(-3x0+4x1-x2)/(2h)={fmt(v_fd)},  a0≈(x0-2x1+x2)/h^2={fmt(a_fd)}")
        elif i == len(ts)-1:
            v_fd = (3*xs[-1] - 4*xs[-2] + xs[-3])/(2*h)
            a_fd = (xs[-1] - 2*xs[-2] + xs[-3])/(h*h)
            L.append(f"    (malla uniforme h={fmt(h)}) DF: vN≈(3xN-4xN-1+xN-2)/(2h)={fmt(v_fd)},  aN≈(xN-2xN-1+xN-2)/h^2={fmt(a_fd)}")
        else:
            v_fd = (xs[i+1] - xs[i-1])/(2*h)
            a_fd = (xs[i+1] - 2*xs[i] + xs[i-1])/(h*h)
            L.append(f"    (malla uniforme h={fmt(h)}) DF: vi≈(x(i+1)-x(i-1))/(2h)={fmt(v_fd)},  ai≈(x(i+1)-2xi+x(i-1))/h^2={fmt(a_fd)}")

    return L



def _format_step_block(i, ts, xs, vi, ai, idx, A, bvec, coeffs, ti, method, uniform_h):
    """
    Devuelve un bloque de líneas de texto explicando:
    - Ventana usada
    - Sistema A·[a,b,c]^T = b
    - Coeficientes a,b,c
    - Cálculo de v(t_i) y a(t_i)
    - (si hay malla uniforme) comparación con fórmulas de diferencias finitas
    """
    a, b, c = coeffs
    (t0, t1, t2) = (float(ts[idx[0]]), float(ts[idx[1]]), float(ts[idx[2]]))
    (x0, x1, x2) = (float(xs[idx[0]]), float(xs[idx[1]]), float(xs[idx[2]]))

    L = []
    L.append(f"{i:>2}.  método = {method}")
    L.append(f"    Ventana usada: ({t0:g},{x0:g}), ({t1:g},{x1:g}), ({t2:g},{x2:g})")
    L.append("    Ajuste cuadrático:  p(t)=a t² + b t + c  que pasa por los 3 puntos")
    L.append("    Sistema A·[a,b,c]^T = b  con")
    L.append(f"       A = [[{t0**2:g}, {t0:g}, 1], [{t1**2:g}, {t1:g}, 1], [{t2**2:g}, {t2:g}, 1]]")
    L.append(f"       b = [{x0:g}, {x1:g}, {x2:g}]")
    L.append(f"    ⇒ (a,b,c) = ({a:g}, {b:g}, {c:g})")
    L.append(f"    v(t_i)=p'(t_i)=2a·t_i+b = 2·{a:g}·{ti:g} + {b:g} = {vi:g}")
    L.append(f"    a(t_i)=p''(t_i)=2a = {2*a:g}")

    # Si la malla es uniforme, muestro la fórmula clásica equivalente
    is_uniform, h = uniform_h
    if is_uniform:
        # índices relativos para fórmulas FD
        n = len(ts)
        if i == 0:  # adelante
            v_fd = (-3*xs[0] + 4*xs[1] - xs[2])/(2*h)
            a_fd = (xs[0] - 2*xs[1] + xs[2])/(h*h)
            L.append(f"    (malla uniforme h={h:g}) fórmulas DF: v₀≈(-3x₀+4x₁-x₂)/(2h)={v_fd:g},  a₀≈(x₀-2x₁+x₂)/h²={a_fd:g}")
        elif i == n-1:  # atrás
            v_fd = (3*xs[-1] - 4*xs[-2] + xs[-3])/(2*h)
            a_fd = (xs[-1] - 2*xs[-2] + xs[-3])/(h*h)
            L.append(
                f"    (malla uniforme h={h:g}) fórmulas DF: "
                f"v_{n-1}≈(3x_{n-1}-4x_{n-2}+x_{n-3})/(2h)={v_fd:g},  "
                f"a_{n-1}≈(x_{n-1}-2x_{n-2}+x_{n-3})/h²={a_fd:g}"
            )
        else:  # central
            v_fd = (xs[i+1] - xs[i-1])/(2*h)
            a_fd = (xs[i+1] - 2*xs[i] + xs[i-1])/(h*h)
            L.append(f"    (malla uniforme h={h:g}) fórmulas DF: v_i≈(x_{i+1}-x_{i-1})/(2h)={v_fd:g},  a_i≈(x_{i+1}-2x_i+x_{i-1})/h²={a_fd:g}")
    return L

def tabla_vel_acc(ts, xs):
    """
    Devuelve t ordenado, x ordenado, v, a y el tipo de esquema usado en cada i.
    """
    n = len(ts)
    if n < 3:
        raise ValueError("Se necesitan al menos 3 puntos (t,x) para estimar v y a.")
    orden = np.argsort(ts)
    ts = [ts[k] for k in orden]
    xs = [xs[k] for k in orden]

    v = np.zeros(n, float)
    a = np.zeros(n, float)
    metodo = []

    for i in range(n):
        vi, ai, idx, *_ = _quad_local_derivs_steps(ts, xs, i)
        v[i] = vi; a[i] = ai
        if idx == [i-1, i, i+1]:
            metodo.append("central")
        elif i == 0:
            metodo.append("adelante (unilateral)")
        else:
            metodo.append("atrás (unilateral)")
    return ts, xs, v, a, metodo


# ================ Ventana principal ================
class DiferenciasFinitasWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diferencias finitas — métodos prog/regr/cent + LaTeX + errores")
        self.setGeometry(100, 100, 1200, 760)

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Panel izquierdo
        left = QVBoxLayout(); root.addLayout(left, 1)

        # --- Modo función f(x) ---
        left.addWidget(QLabel("Modo f(x):"))
        row1 = QHBoxLayout(); left.addLayout(row1)
        row1.addWidget(QLabel("f(x):")); self.fx_input = QLineEdit("sin(x)"); row1.addWidget(self.fx_input)
        row2 = QHBoxLayout(); left.addLayout(row2)
        row2.addWidget(QLabel("x₀:")); self.x0_input = QLineEdit("0.5"); row2.addWidget(self.x0_input)
        row2.addWidget(QLabel("h:"));  self.h_input  = QLineEdit("0.1");  row2.addWidget(self.h_input)

        left.addWidget(QLabel("Órdenes a calcular (siempre prog/regr/cent):"))
        row3 = QHBoxLayout(); left.addLayout(row3)
        self.chk1 = QCheckBox("1ª"); self.chk1.setChecked(True); row3.addWidget(self.chk1)
        self.chk2 = QCheckBox("2ª"); row3.addWidget(self.chk2)
        self.chk3 = QCheckBox("3ª"); row3.addWidget(self.chk3)
        row3.addStretch(1)
        self.chk_o2 = QCheckBox("Adel/atrás 2º orden (1ª deriv.)")
        left.addWidget(self.chk_o2)

        self.btn_calc = QPushButton("Calcular en x₀")
        style_primary_button(self.btn_calc)
        self.btn_calc.clicked.connect(self.calcular)
        left.addWidget(self.btn_calc)

        # --- Modo tabla t, x(t) ---
        left.addSpacing(8)
        left.addWidget(QLabel("Modo tabla (t y x(t) separados por comas):"))
        self.t_input = QLineEdit("0,2,4,6,8,10,12,14,16")                 # ejemplo ej.7
        self.x_input = QLineEdit("0,0.7,1.8,3.4,5.1,6.3,7.3,8.0,8.4")     # ejemplo ej.7
        left.addWidget(QLabel("t (seg):")); left.addWidget(self.t_input)
        left.addWidget(QLabel("x(t) (m):")); left.addWidget(self.x_input)

        self.btn_tabla = QPushButton("Calcular v(t), a(t) desde tabla")
        style_primary_button(self.btn_tabla)
        self.btn_tabla.clicked.connect(self.calcular_tabla)
        left.addWidget(self.btn_tabla)

        left.addStretch(1)
        


        # Panel derecho
        right = QVBoxLayout(); root.addLayout(right, 2)
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        right.addWidget(self.canvas)

        # 👉 eje derecho persistente (lo vamos a “resetear” cada vez)
        self.ax2 = None

        self.output = QTextEdit(); self.output.setReadOnly(True)
        right.addWidget(self.output)

    # --------------- modo f(x) ---------------
    def calcular(self):
        try:
            expr_fx = normalizar_expr(self.fx_input.text())
            f = crear_funcion_con_lhopital(expr_fx)

            x = sp.symbols("x")
            fx_sym = sp.simplify(sp.sympify(expr_fx, locals={"x": x, "pi": sp.pi, "E": sp.E}))

            x0 = float(sp.sympify(normalizar_expr(self.x0_input.text()), locals={"pi": sp.pi, "E": sp.E}))
            h  = float(sp.sympify(normalizar_expr(self.h_input.text()),  locals={"pi": sp.pi, "E": sp.E}))

            orders = []
            if self.chk1.isChecked(): orders.append(1)
            if self.chk2.isChecked(): orders.append(2)
            if self.chk3.isChecked(): orders.append(3)
            if not orders: orders = [1]

            derivs_sym = {n: sp.diff(fx_sym, x, n) for n in sorted(set(orders))}

            self.output.clear()
            self.output.append("=== Función y derivadas simbólicas ===")
            self.output.append(f"f(x) = {sp.sstr(fx_sym)}")
            for n in sorted(derivs_sym.keys()):
                self.output.append(f"f^{n}(x) = {sp.sstr(derivs_sym[n])}")
            self.output.append("")

            latex_left_lines  = [fr"f(x)= {sp.latex(fx_sym)}"]
            for n in sorted(derivs_sym.keys()):
                latex_left_lines.append(fr"f^{{({n})}}(x)= {sp.latex(derivs_sym[n])}")
            latex_right_lines = []

            for n in orders:
                if n == 1:
                    approx, pasos = fd_first(f, x0, h, o2_forward_backward=self.chk_o2.isChecked())
                elif n == 2:
                    approx, pasos = fd_second(f, x0, h)
                else:
                    approx, pasos = fd_third(f, x0, h)

                self.output.append(f"—  {n}ª derivada  —  (progresiva, regresiva, central)")
                for p in pasos: self.output.append(p)

                exact = None
                if derivs_sym.get(n) is not None:
                    try:
                        exact = float(sp.N(derivs_sym[n].subs(x, x0)))
                        self.output.append(f"Exacto: f^{n}({fmt(x0)}) = {fmt(exact)}")
                        self.output.append(
                            "Errores absolutos — "
                            + f"|prog-true|={fmt(abs(approx['prog']-exact))}, "
                            + f"|regr-true|={fmt(abs(approx['regr']-exact))}, "
                            + f"|cent-true|={fmt(abs(approx['cent']-exact))}"
                        )
                    except Exception:
                        pass
                self.output.append("")

                if n == 1:
                    latex_right_lines += [
                        fr"f'({fmt(x0)})_{{prog}}\approx {fmt(approx['prog'])}",
                        fr"f'({fmt(x0)})_{{regr}}\approx {fmt(approx['regr'])}",
                        fr"f'({fmt(x0)})_{{cent}}\approx {fmt(approx['cent'])}",
                    ]
                    if exact is not None: latex_right_lines.append(fr"f'({fmt(x0)})= {fmt(exact)}")
                elif n == 2:
                    latex_right_lines += [
                        fr"f''({fmt(x0)})_{{prog}}\approx {fmt(approx['prog'])}",
                        fr"f''({fmt(x0)})_{{regr}}\approx {fmt(approx['regr'])}",
                        fr"f''({fmt(x0)})_{{cent}}\approx {fmt(approx['cent'])}",
                    ]
                    if exact is not None: latex_right_lines.append(fr"f''({fmt(x0)})= {fmt(exact)}")
                else:
                    latex_right_lines += [
                        fr"f^{{(3)}}({fmt(x0)})_{{prog}}\approx {fmt(approx['prog'])}",
                        fr"f^{{(3)}}({fmt(x0)})_{{regr}}\approx {fmt(approx['regr'])}",
                        fr"f^{{(3)}}({fmt(x0)})_{{cent}}\approx {fmt(approx['cent'])}",
                    ]
                    if exact is not None: latex_right_lines.append(fr"f^{{(3)}}({fmt(x0)})= {fmt(exact)}")

            # --- gráfico modo f(x) ---
            self.ax.clear()
            # si había un eje derecho previo, lo removemos:
            if self.ax2 is not None:
                try: self.ax2.remove()
                except Exception: pass
            self.ax2 = None

            X = np.linspace(x0 - 2*abs(h) - 1, x0 + 2*abs(h) + 1, 400)
            Y = np.array([f(xx) for xx in X], dtype=float)
            self.ax.plot(X, Y, label="f(x)")
            self.ax.plot([x0], [f(x0)], "ro", label="(x₀, f(x₀))")

            if 1 in orders:
                approx_1, _ = fd_first(f, x0, h)
                m = approx_1["cent"]
                self.ax.plot(X, (f(x0) + m*(X - x0)), "g--", label="Tangente (1ª central)")

            self.ax.set_title("Diferencias finitas (progresiva, regresiva, central)")
            self.ax.grid(True)
            self.ax.legend(loc="lower right", framealpha=0.9, facecolor="white", edgecolor="black")

            try:
                self.ax.text(0.02, 0.98, mt_block(latex_left_lines),
                             transform=self.ax.transAxes, va="top", ha="left",
                             fontsize=11, bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))
            except Exception:
                self.ax.text(0.02, 0.98, "\n".join(latex_left_lines),
                             transform=self.ax.transAxes, va="top", ha="left",
                             fontsize=10, bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))

            try:
                if latex_right_lines:
                    self.ax.text(0.98, 0.98, mt_block(latex_right_lines),
                                 transform=self.ax.transAxes, va="top", ha="right",
                                 fontsize=11, bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))
            except Exception:
                self.ax.text(0.98, 0.98, "\n".join(latex_right_lines),
                             transform=self.ax.transAxes, va="top", ha="right",
                             fontsize=10, bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))

            self.figure.tight_layout(pad=1.05)
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # --------------- MODO TABLA t, x(t) ---------------
    def calcular_tabla(self):
        try:
            ts = _parse_list(self.t_input.text())
            xs = _parse_list(self.x_input.text())
            if len(ts) != len(xs):
                raise ValueError("Las listas t y x(t) deben tener la misma longitud.")
            if len(ts) < 3:
                raise ValueError("Se necesitan al menos 3 puntos (t,x).")

            # ordenar por t
            order = np.argsort(ts)
            ts = [ts[k] for k in order]
            xs = [xs[k] for k in order]

            # ¿malla uniforme?
            uniform_h = _uniform_step(ts)

            n = len(ts)
            v = np.zeros(n, float)
            a = np.zeros(n, float)
            metodo = []

            # calculo v, a y método por fila
            for i in range(n):
                vi, ai, idx, *_ = _quad_local_derivs_steps(ts, xs, i)
                v[i] = vi; a[i] = ai
                if idx == [i-1, i, i+1]:
                    metodo.append("central")
                elif i == 0:
                    metodo.append("adelante (unilateral)")
                else:
                    metodo.append("atrás (unilateral)")

            # ============ SALIDA ============ 
            self.output.clear()
            self.output.append("=== Modo tabla: velocidades y aceleraciones ===")
            self.output.append("Método: ajuste cuadrático local (central en interiores; unilaterales en extremos).")
            if uniform_h[0]:
                self.output.append(f"Detección: malla uniforme con h ≈ {uniform_h[1]:g}\n")
            else:
                self.output.append("Detección: malla NO uniforme\n")

            # 1) Tabla compacta completa
            header = f"{'i':>2}  {'t':>8}  {'x(t)':>12}  {'v(t)':>12}  {'a(t)':>12}   método"
            self.output.append(header)
            for i in range(n):
                self.output.append(
                    f"{i:>2}  {fmt(ts[i]):>8}  {fmt(xs[i]):>12}  {fmt(v[i]):>12}  {fmt(a[i]):>12}   {metodo[i]}"
                )

            # 2) Paso a paso por fila con despeje explícito
            self.output.append("\n=== Paso a paso (ajuste cuadrático local con despeje de a,b,c) ===")
            for i in range(n):
                vi, ai, idx, *_tmp = _quad_local_derivs_steps(ts, xs, i)
                if idx == [i-1, i, i+1]:
                    method = "central"
                elif i == 0:
                    method = "adelante (unilateral)"
                else:
                    method = "atrás (unilateral)"
                blk = _explain_step_block(i, ts, xs, vi, ai, idx, ts[i], method, uniform_h)
                for line in blk:
                    self.output.append(line)
                self.output.append("")

            # ===== Gráfico =====
            self.ax.clear()
            if self.ax2 is not None:
                try: self.ax2.remove()
                except Exception: pass
            self.ax2 = None

            self.ax.set_title("Tabla: posición, velocidad y aceleración")
            self.ax.set_xlabel("t"); self.ax.set_ylabel("x(t)")
            ln1 = self.ax.plot(ts, xs, "o-", label="x(t)")

            self.ax2 = self.ax.twinx()
            self.ax2.set_ylabel("v(t), a(t)")
            ln2 = self.ax2.plot(ts, v, "--o", label="v(t)")
            ln3 = self.ax2.plot(ts, a, ":s", label="a(t)")

            yRight = np.concatenate([np.asarray(v, float), np.asarray(a, float)])
            ymin, ymax = float(np.min(yRight)), float(np.max(yRight))
            if abs(ymax - ymin) < 1e-12:
                ymin -= 1.0; ymax += 1.0
            pad = 0.05*(ymax - ymin)
            self.ax2.set_ylim(ymin - pad, ymax + pad)

            lns = ln1 + ln2 + ln3
            labs = [l.get_label() for l in lns]
            self.ax.legend(lns, labs, loc="best", framealpha=0.9)

            self.ax.grid(True)
            self.figure.tight_layout(pad=1.05)
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

            
            


# ================ main ================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DiferenciasFinitasWindow()
    win.show()
    sys.exit(app.exec())
