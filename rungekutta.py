# -*- coding: utf-8 -*-
"""
EcuacionesDiferenciales.py
EDOs de 1er orden: Euler, Heun, RK4.
- Paso a paso (estilo apunte) + Error por paso y ε final (si hay analítica).
- Solución analítica automática (lineal a(t)*y + b(t) con factor integrante;
  si no, intenta dsolve() con y(t0)=y0).
- Gráfico automático; botón "Graficar TODOS".
- n = (b-a)/h calculado automáticamente.
"""
import matplotlib
matplotlib.use("QtAgg")


from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QComboBox, QHBoxLayout, QMessageBox, QSplitter, QCheckBox
)
import sympy as sp
from sympy import Function, Eq, dsolve
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use("QtAgg")

# debajo de imports, cerca de los utils

_NUM_NS = {'pi': sp.pi, 'e': sp.E, 'E': sp.E}

def _parse_num(text: str) -> float:
    """
    Acepta '3.14', 'pi', 'pi/10', '2*pi', 'e^2', etc.
    Devuelve float o lanza ValueError si no se puede.
    """
    s = text.strip().replace('^', '**')
    try:
        val = float(sp.N(sp.sympify(s, locals=_NUM_NS)))
        if np.isnan(val):
            raise ValueError
        return val
    except Exception:
        raise ValueError(f"Valor numérico inválido: {text!r}")



# ---------------- Utils ----------------

def _build_f_t_y(expr_str: str):
    """Devuelve t, y (símbolos), f(t,y) (sympy) y f_num(t,y) (num)."""
    t, y = sp.symbols('t y', real=True)
    safe_ns = {
        't': t, 'y': y,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
        'E': sp.E, 'pi': sp.pi, 'abs': sp.Abs
    }
    s = expr_str.replace('^', '**')
    try:
        f = sp.simplify(sp.sympify(s, locals=safe_ns))
    except Exception as e:
        raise ValueError(f"Expresión inválida para f(t,y): {e}")
    try:
        f_num = sp.lambdify((t, y), f, 'math')
    except Exception as e:
        raise ValueError(f"No se pudo crear función numérica: {e}")
    return t, y, f, f_num

def _float(x):
    """Convierte a float (evita NaN tanto como sea posible)."""
    try:
        v = float(x)
        if np.isnan(v):
            raise ValueError
        return v
    except Exception:
        try:
            v = float(sp.N(x, 50))
            if np.isnan(v):
                raise ValueError
            return v
        except Exception:
            return 0.0  # último recurso para que la tabla no muestre 'nan'

# ---- Solución analítica automática ----

def _solve_exact_linear(f_sym, t0, y0):
    """
    Si f(t,y) = a(t)*y + b(t) (afín en y), resuelve con factor integrante:
        μ(t)=exp(∫a(t)dt), y(t) = μ(t)*( y0 + ∫_{t0}^t b(s)/μ(s) ds )
    Devuelve (expr, func) o (None, None) si no es lineal.
    """
    t, y = sp.symbols('t y', real=True)
    try:
        a_t = sp.simplify(sp.diff(f_sym, y))
        b_t = sp.simplify(f_sym - a_t*y)
        if y in a_t.free_symbols or y in b_t.free_symbols:
            return None, None
        mu = sp.exp(sp.integrate(a_t, (t, t0, t)))
        integrando = sp.simplify(b_t / mu)
        I = sp.integrate(integrando, (t, t0, t))
        y_expr = sp.simplify(mu*(y0 + I))
        y_num = sp.lambdify(sp.Symbol('t'), y_expr, 'math')
        _ = _float(y_num(t0))
        return y_expr, y_num
    except Exception:
        return None, None

def _solve_exact(f_sym, t0, y0):
    """
    Primero intenta lineal en y; si no aplica, usa dsolve(Y' = f(t,Y), ics={Y(t0)=y0})
    sustituyendo y -> Y(t).
    """
    y_expr, y_num = _solve_exact_linear(f_sym, t0, y0)
    if y_expr is not None:
        return y_expr, y_num

    t = sp.Symbol('t', real=True)
    Y = Function('y')
    try:
        f_for_dsolve = f_sym
        if sp.Symbol('y') in f_for_dsolve.free_symbols:
            f_for_dsolve = f_for_dsolve.subs({sp.Symbol('y'): Y(t)})
        ode = Eq(sp.diff(Y(t), t), f_for_dsolve)
        sol = dsolve(ode, ics={Y(t0): y0})
        rhs = sol.rhs if isinstance(sol, sp.Eq) else (sol.rhs if hasattr(sol, "rhs") else sol)
        rhs = sp.simplify(rhs)
        y_num = sp.lambdify(t, rhs, 'math')
        _ = _float(y_num(t0))
        return rhs, y_num
    except Exception:
        return None, None

# ---- Formato de tablas ----
def _fmt_header(cols, widths):
    return "  ".join(f"{str(c):>{w}s}" for c, w in zip(cols, widths))

def _fmt_row(values, widths, fmts):
    parts = []
    for v, w, f in zip(values, widths, fmts):
        if isinstance(v, (int, np.integer)) and f == "d":
            parts.append(f"{v:{w}d}")
        elif isinstance(v, (float, np.floating)) and f != "s":
            parts.append(("{:"+str(w)+f+"}").format(float(v)))
        else:
            parts.append(f"{str(v):>{w}s}")
    return "  ".join(parts)


# ---------------- Métodos con logs + error ----------------

def euler(f_num, t0, y0, h, n, y_exact=None):
    ts = [t0]; ys = [y0]; log = []
    log += ["Método de Euler (1° orden)",
            "Recurrencia:  y_{n+1} = y_n + h · f(t_n, y_n)",
            f"h = {h}\n",
            "Tabla:"]
    widths = [2, 8, 12, 12, 14]
    fmts   = ["d", ".6f", ".6f", ".6f", ".10f"]
    header_cols = ["n","t_n","y_n","y_{n+1}"] + (["Error"] if y_exact else [])
    log.append(_fmt_header(header_cols, widths[:len(header_cols)]))

    t, y = float(t0), float(y0)
    eps_final = None
    for i in range(n):
        fval = _float(f_num(t, y))
        y_next = _float(y + h*fval)
        t_next = _float(t + h)
        if y_exact is None:
            row = _fmt_row([i, t, y, y_next], widths[:4], fmts[:4])
        else:
            err = abs(_float(y_exact(t_next) - y_next)); eps_final = err
            row = _fmt_row([i, t, y, y_next, err], widths, fmts)
        log.append(row)
        ts.append(t_next); ys.append(y_next)
        t, y = t_next, y_next
    return ys, ts, "\n".join(log), eps_final

def heun(f_num, t0, y0, h, n, y_exact=None):
    ts = [t0]; ys = [y0]; log = []
    log += ["Método de Heun (2° orden) — Euler mejorado",
            "Pred.: y*_{n+1} = y_n + h · f(t_n, y_n)",
            "Corr.: y_{n+1} = y_n + (h/2) · [ f(t_n,y_n) + f(t_{n+1}, y*_{n+1}) ]",
            f"h = {h}\n",
            "Detalle por paso:"]
    widths = [2, 8, 12, 12, 12, 14]
    fmts   = ["d", ".6f", ".6f", ".6f", ".6f", ".10f"]
    header_cols = ["n","t_n","y_n","y*_{n+1}","y_{n+1}"] + (["Error"] if y_exact else [])
    log.append(_fmt_header(header_cols, widths[:len(header_cols)]))

    t, y = float(t0), float(y0)
    eps_final = None
    for i in range(n):
        t_next = _float(t + h)
        f_n = _float(f_num(t, y))
        y_star = _float(y + h*f_n)
        f_next = _float(f_num(t_next, y_star))
        y_next = _float(y + (h/2.0)*(f_n + f_next))
        if y_exact is None:
            row = _fmt_row([i, t, y, y_star, y_next], widths[:5], fmts[:5])
        else:
            err = abs(_float(y_exact(t_next) - y_next)); eps_final = err
            row = _fmt_row([i, t, y, y_star, y_next, err], widths, fmts)
        log.append(row)
        ts.append(t_next); ys.append(y_next)
        t, y = t_next, y_next
    return ys, ts, "\n".join(log), eps_final

def rk4(f_num, t0, y0, h, n, y_exact=None):
    ts = [t0]; ys = [y0]; log = []
    log += ["Runge-Kutta 4 (4° orden) — muy preciso",
            "y_{n+1} = y_n + (h/6)·(k1 + 2·k2 + 2·k3 + k4)",
            "k1=f(t_n,y_n); k2=f(t_n+h/2, y_n+h k1/2); k3=f(t_n+h/2, y_n+h k2/2); k4=f(t_n+h, y_n+h k3)",
            f"h = {h}\n",
            "Detalle por paso:"]
    widths = [2, 8, 11, 11, 11, 11, 12, 14, 14]
    fmts   = ["d",".6f",".6f",".6f",".6f",".6f",".6f",".6f",".10f"]
    header_cols = ["n","t_n","y_n","k1","k2","k3","k4","y_{n+1}"] + (["Error"] if y_exact else [])
    log.append(_fmt_header(header_cols, widths[:len(header_cols)]))

    t, y = float(t0), float(y0)
    eps_final = None
    for i in range(n):
        k1 = _float(f_num(t, y))
        k2 = _float(f_num(t + h/2.0, y + (h/2.0)*k1))
        k3 = _float(f_num(t + h/2.0, y + (h/2.0)*k2))
        k4 = _float(f_num(t + h,     y + h*k3))
        y_next = _float(y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4))
        t_next = _float(t + h)
        if y_exact is None:
            row = _fmt_row([i, t, y, k1, k2, k3, k4, y_next], widths[:8], fmts[:8])
        else:
            err = abs(_float(y_exact(t_next) - y_next)); eps_final = err
            row = _fmt_row([i, t, y, k1, k2, k3, k4, y_next, err], widths, fmts)
        log.append(row)
        ys.append(y_next); ts.append(t_next)
        t, y = t_next, y_next
    return ys, ts, "\n".join(log), eps_final


# ---------------- UI ----------------

class EDOsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EDOs: Euler, Heun y Runge-Kutta 4")
        self.setMinimumWidth(980)
        self.setMinimumHeight(640)

        vmain = QVBoxLayout(self)
        top = QGridLayout(); vmain.addLayout(top)

        # Entradas: f, t0, y0, a, b, h
        row = 0
        top.addWidget(QLabel("f(t, y) ="), row, 0)
        self.edit_f = QLineEdit("t - y"); top.addWidget(self.edit_f, row, 1, 1, 3)

        row += 1
        top.addWidget(QLabel("t₀"), row, 0)
        self.edit_t0 = QLineEdit("0"); top.addWidget(self.edit_t0, row, 1)
        top.addWidget(QLabel("y₀"), row, 2)
        self.edit_y0 = QLineEdit("1"); top.addWidget(self.edit_y0, row, 3)

        row += 1
        top.addWidget(QLabel("a"), row, 0)
        self.edit_a = QLineEdit("0"); top.addWidget(self.edit_a, row, 1)
        top.addWidget(QLabel("b"), row, 2)
        self.edit_b = QLineEdit("1"); top.addWidget(self.edit_b, row, 3)

        row += 1
        top.addWidget(QLabel("h (paso)"), row, 0)
        self.edit_h = QLineEdit("0.1"); top.addWidget(self.edit_h, row, 1)

        row += 1
        top.addWidget(QLabel("Método (para el paso a paso)"), row, 0)
        self.combo_method = QComboBox()
        shead = ["Euler (1°)", "Heun (2°)", "Runge-Kutta 4 (4°)"]
        self.combo_method.addItems(shead)
        top.addWidget(self.combo_method, row, 1)
        
        # Checkbox de campo direccional
        self.chk_slope = QCheckBox("Mostrar campos direccionales (isoclinas)")
        self.chk_slope.setChecked(False)  # por defecto apagado

        # Botones
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Calcular")
        self.btn_plot_all = QPushButton("Graficar TODOS")
        btn_row.addWidget(self.btn_run);
        btn_row.addWidget(self.chk_slope);
        btn_row.addStretch(1); btn_row.addWidget(self.btn_plot_all)
        vmain.addLayout(btn_row)

        # Splitter texto / gráfico
        splitter = QSplitter(Qt.Orientation.Horizontal); vmain.addWidget(splitter, stretch=1)

        self.out = QTextEdit(); self.out.setReadOnly(True)
        self.out.setStyleSheet("font-family: 'JetBrains Mono','Consolas', monospace; font-size: 13px; padding: 6px;")
        splitter.addWidget(self.out)

        self.fig, self.ax = plt.subplots(); self.canvas = FigureCanvas(self.fig)
        splitter.addWidget(self.canvas); splitter.setSizes([620, 360])

        # Acciones
        self.btn_run.clicked.connect(self._run)
        self.btn_plot_all.clicked.connect(self._plot_all)

        # Tema claro
        self.setStyleSheet("""
            QWidget { background: #ffffff; color: #0f172a; }
            QLabel { font-weight: 600; color: #111827; }
            QLineEdit, QComboBox, QTextEdit {
                background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 6px;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus { border: 1px solid #2563eb; }
            QPushButton { background: #2563eb; color: white; border: none; border-radius: 8px; padding: 8px 14px; font-weight: 600; }
            QPushButton:hover { background: #1d4ed8; }
            QPushButton:pressed { background: #1e40af; }
        """)

        # Estado
        self._last_ts = None; self._last_ys = None
        self._last_y_exact = None; self._last_title = ""
        self._f_sym = None; self._f_num = None
        self._t0 = 0.0; self._y0 = 0.0; self._h = 0.1; self._n = 10; self._a = 0.0; self._b = 0.0

    # -------- helpers --------
    def _read_params(self):
        try:
            t0 = _parse_num(self.edit_t0.text())
            y0 = _parse_num(self.edit_y0.text())
            a  = _parse_num(self.edit_a.text())
            b  = _parse_num(self.edit_b.text())
            h  = _parse_num(self.edit_h.text())
            if h <= 0:
                raise ValueError("h debe ser > 0.")
        except Exception as e:
            raise ValueError("t0, y0, a, b, h deben ser números/expresiones válidas; h>0. "
                             f"Detalle: {e}")

        n_exact = (b - a) / h
        n = int(round(n_exact))
        if abs(n - n_exact) > 1e-10:
            QMessageBox.information(
                self, "Aviso",
                f"(b-a)/h = {n_exact:.6g} no es entero. Uso n={n} (ajustado). "
                "Podés ajustar h o b para que sea exacto."
            )
        if n <= 0:
            raise ValueError("n calculado no positivo. Verificá [a,b] y h.")
        return t0, y0, h, n, a, b


    # -------- callbacks --------
    def _run(self):
        try:
            _, _, f_sym, f_num = _build_f_t_y(self.edit_f.text().strip())
        except ValueError as e:
            QMessageBox.critical(self, "Error en f(t,y)", str(e)); return

        try:
            t0, y0, h, n, a, b = self._read_params()
        except ValueError as e:
            QMessageBox.critical(self, "Parámetros inválidos", str(e)); return

        # solución exacta automática
        y_exact_expr, y_exact = _solve_exact(f_sym, t0, y0)

        # método elegido
        method_name = self.combo_method.currentText()
        if "Euler" in method_name and "1" in method_name:
            ys, ts, log, eps = euler(f_num, t0, y0, h, n, y_exact)
        elif "Heun" in method_name:
            ys, ts, log, eps = heun(f_num, t0, y0, h, n, y_exact)
        else:
            ys, ts, log, eps = rk4(f_num, t0, y0, h, n, y_exact)

        # encabezado + reporte
        report = []
        report.append("===============================================")
        report.append(f"EDO: y' = f(t,y) con f(t,y) = {sp.sstr(f_sym)}")
        report.append(f"t0 = {t0}, y0 = {y0}, h = {h}, intervalo [a,b] = [{a}, {b}]")
        report.append(f"n (calculado) = (b-a)/h = {(b-a)/h:.6g} ⇒ n = {n}")
        report.append("===============================================\n")
        report.append(log)

        if y_exact_expr is not None:
            report.append("\n--- Solución analítica (automática) ---")
            report.append(f"y(t) = {sp.sstr(y_exact_expr)}")
            t_final = ts[-1]
            y_anal_final = _float(y_exact(t_final))
            y_num_final  = _float(ys[-1])
            eps_final = abs(y_anal_final - y_num_final)
            report.append(f"t_final = {t_final:.6g}")
            report.append(f"y_num(t_final) = {y_num_final:.9g}")
            report.append(f"y_analítica(t_final) = {y_anal_final:.9g}")
            report.append(f"ε final = |y_a - y_n| = {eps_final:.9g}")
            if eps is not None:
                report.append(f"(último error de tabla) = {eps:.9g}")
        else:
            report.append("\n(No fue posible obtener solución analítica cerrada)")

        self.out.setPlainText("\n".join(report))

        # guardar y graficar
        self._last_ts = ts; self._last_ys = ys
        self._last_y_exact = y_exact; self._last_title = method_name
        self._f_sym = f_sym; self._f_num = f_num
        self._t0, self._y0, self._h, self._n, self._a, self._b = t0, y0, h, n, a, b

        self._plot_last()

    def _plot_last(self):
        if self._last_ts is None: return
        self.ax.clear()
        # Si está tildado, dibujar campo direccional de fondo
        if self.chk_slope.isChecked() and self._f_num is not None:
            # rango en t: del último gráfico (o [a,b] si existe)
            t_min = self._last_ts[0]; t_max = self._last_ts[-1]
            if hasattr(self, "_a") and hasattr(self, "_b") and self._a != self._b:
                t_min, t_max = self._a, self._b

            # rango en y: en base a la solución numérica con un margen
            y_min, y_max = min(self._last_ys), max(self._last_ys)
            pad = 0.2*(y_max - y_min + 1e-9)
            y_min -= pad; y_max += pad

            self._draw_slope_field(self.ax, self._f_num, t_min, t_max, y_min, y_max)
        self.ax.plot(self._last_ts, self._last_ys, marker='o', linestyle='-', label=self._last_title)
        if self._last_y_exact is not None:
            t_min, t_max = self._last_ts[0], self._last_ts[-1]
            ts_dense = np.linspace(t_min, t_max, max(120, 12*len(self._last_ts)))
            ys_dense = [ _float(self._last_y_exact(tt)) for tt in ts_dense ]
            self.ax.plot(ts_dense, ys_dense, linestyle='--', label='Analítica')
        self.ax.set_xlabel('t'); self.ax.set_ylabel('y(t)')
        self.ax.set_title(self._last_title)
        self.ax.grid(True, linestyle=':'); self.ax.legend()
        self.canvas.draw_idle()

    def _plot_all(self):
        if self._f_num is None:
            self._run()
            if self._f_num is None: return

        self.ax.clear()
       

        if self.chk_slope.isChecked() and self._f_num is not None:
            t_min, t_max = self._a, self._b
            # si por alguna razón no están seteados, usá el rango del último cálculo
            if t_min == t_max:
                t_min = self._t0
                t_max = self._t0 + self._h*self._n
            # y rango y a partir de una de las trayectorias (RK4, por ejemplo)
            ys_r, ts_r, _, _ = rk4(self._f_num, self._t0, self._y0, self._h, self._n, None)
            y_min, y_max = min(ys_r), max(ys_r)
            pad = 0.2*(y_max - y_min + 1e-9)
            self._draw_slope_field(self.ax, self._f_num, t_min, t_max, y_min - pad, y_max + pad)

        ys_e, ts_e, _, _ = euler(self._f_num, self._t0, self._y0, self._h, self._n, self._last_y_exact)
        self.ax.plot(ts_e, ys_e, marker='o', linestyle='-', label='Euler')
        ys_h, ts_h, _, _ = heun(self._f_num, self._t0, self._y0, self._h, self._n, self._last_y_exact)
        self.ax.plot(ts_h, ys_h, marker='s', linestyle='-', label='Heun')
        ys_r, ts_r, _, _ = rk4(self._f_num, self._t0, self._y0, self._h, self._n, self._last_y_exact)
        self.ax.plot(ts_r, ys_r, marker='^', linestyle='-', label='RK4')

        if self._last_y_exact is not None:
            ts_dense = np.linspace(self._a, self._b, max(200, 15*self._n))
            ys_dense = [ _float(self._last_y_exact(tt)) for tt in ts_dense ]
            self.ax.plot(ts_dense, ys_dense, linestyle='--', label='Analítica')

        self.ax.set_xlabel('t'); self.ax.set_ylabel('y(t)')
        self.ax.set_title(f"Comparación de métodos — f(t,y)={sp.sstr(self._f_sym)}")
        self.ax.grid(True, linestyle=':'); self.ax.legend()
        self.canvas.draw_idle()
        
    def _draw_slope_field(self, ax, f_num, a, b, y_min, y_max, density=25, n_levels=7):
        """
        Dibuja campo direccional (quiver) y curvas isoclinas (contour) de f(t,y).
        - a,b: rango en t
        - y_min,y_max: rango en y
        """
        try:
            T = np.linspace(a, b, density)
            Y = np.linspace(y_min, y_max, density)
            TT, YY = np.meshgrid(T, Y)

            def f_safe(tt, yy):
                try:
                    return float(f_num(float(tt), float(yy)))
                except Exception:
                    return np.nan

            F = np.vectorize(f_safe)(TT, YY)

            # Normalizamos para que todas las flechas tengan tamaño similar
            U = np.ones_like(F)   # dt
            V = F                 # dy/dt
            L = np.hypot(U, V)
            L[L == 0] = 1.0
            U /= L; V /= L

            ax.quiver(TT, YY, U, V, angles='xy', width=0.002, alpha=0.45, zorder=1)

            # Isoclinas: niveles entre percentiles 15..85 para evitar outliers
            finite = np.isfinite(F)
            if np.any(finite):
                vmin = np.nanpercentile(F[finite], 15)
                vmax = np.nanpercentile(F[finite], 85)
                if vmin != vmax:
                    levels = np.linspace(vmin, vmax, n_levels)
                    cs = ax.contour(TT, YY, F, levels=levels, linestyles='--', alpha=0.5, zorder=2)
                    ax.clabel(cs, inline=1, fontsize=8)
        except Exception:
            # si algo falla, no rompas el gráfico
            pass

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = EDOsWindow()
    win.show()
    sys.exit(app.exec())
