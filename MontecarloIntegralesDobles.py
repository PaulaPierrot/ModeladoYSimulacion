from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable 

from func_utils import normalizar_expr
from math_keyboard import MathKeyboard

matplotlib.use("QtAgg")

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


# --- Monte Carlo Doble ---
def montecarlo_integral_doble(f, ax, bx, ay, by, n, semilla, conf):
    random.seed(semilla)

    puntos_x, puntos_y, puntos_z = [], [], []
    valores_funcion = []

    for _ in range(n):
        x = random.uniform(ax, bx)
        y = random.uniform(ay, by)
        fx = f(x, y)
        valores_funcion.append(fx)
        puntos_x.append(x)
        puntos_y.append(y)
        puntos_z.append(fx)

    volumen = (bx - ax) * (by - ay)
    promedio = sum(valores_funcion) / len(valores_funcion)
    integral_estimada = volumen * promedio

    varianza = sum((v - promedio) ** 2 for v in valores_funcion) / (n - 1)
    desvio = math.sqrt(varianza)

    error_estandar = volumen * (desvio / math.sqrt(n))

    z = norm.ppf(1 - (1 - conf) / 2)
    intervalo_inf = integral_estimada - z * error_estandar
    intervalo_sup = integral_estimada + z * error_estandar

    return (integral_estimada, (intervalo_inf, intervalo_sup),
            puntos_x, puntos_y, puntos_z, desvio, error_estandar)


class MonteCarloDobleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integración Numérica - Monte Carlo Doble (3D)")
        self.setGeometry(100, 100, 1200, 700)
        self.active_input = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)

        self.fx_input = QLineEdit("x**2+y**2")
        self.ax_input = QLineEdit("0")
        self.bx_input = QLineEdit("1")
        self.ay_input = QLineEdit("0")
        self.by_input = QLineEdit("1")
        self.n_input = QLineEdit("5000")
        self.seed_input = QLineEdit("42")
        self.conf_input = QLineEdit("0.95")

        for entrada in [self.fx_input, self.ax_input, self.bx_input,
                        self.ay_input, self.by_input, self.n_input,
                        self.seed_input, self.conf_input]:
            entrada.installEventFilter(self)

        form_layout.addWidget(QLabel("f(x,y):"))
        form_layout.addWidget(self.fx_input)
        form_layout.addWidget(QLabel("x ∈ ["))
        form_layout.addWidget(self.ax_input)
        form_layout.addWidget(QLabel(",")); form_layout.addWidget(self.bx_input)
        form_layout.addWidget(QLabel("]"))
        form_layout.addWidget(QLabel("y ∈ ["))
        form_layout.addWidget(self.ay_input)
        form_layout.addWidget(QLabel(",")); form_layout.addWidget(self.by_input)
        form_layout.addWidget(QLabel("]"))
        form_layout.addWidget(QLabel("N:"))
        form_layout.addWidget(self.n_input)
        form_layout.addWidget(QLabel("Semilla:"))
        form_layout.addWidget(self.seed_input)
        form_layout.addWidget(QLabel("Confianza:"))
        form_layout.addWidget(self.conf_input)

        self.keyboard = MathKeyboard(self.insertar_texto)
        layout.addWidget(self.keyboard)

        self.calc_btn = QPushButton("Calcular integral Monte Carlo 2D")
        style_primary_button(self.calc_btn) 
        self.calc_btn.clicked.connect(self.calcular)
        layout.addWidget(self.calc_btn)

        # Área de gráfico 
        self.figure = plt.figure(figsize=(8, 5))
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    def eventFilter(self, obj, event):
        if event.type() == event.Type.FocusIn:
            self.active_input = obj
        return super().eventFilter(obj, event)

    def insertar_texto(self, texto):
        if not self.active_input:
            return
        cursor = self.active_input.cursorPosition()
        actual = self.active_input.text()
        nuevo = actual[:cursor] + texto + actual[cursor:]
        self.active_input.setText(nuevo)
        self.active_input.setCursorPosition(cursor + len(texto))

    
    def calcular(self):
        try:
            expr_fx = normalizar_expr(self.fx_input.text())
            expr = sp.sympify(expr_fx)
            vars_ = list(expr.free_symbols)
            f = sp.lambdify(vars_, expr, 'math')

            locals_dict = {"pi": sp.pi, "π": sp.pi, "e": sp.E}
            ax = float(sp.sympify(normalizar_expr(self.ax_input.text()), locals=locals_dict))
            bx = float(sp.sympify(normalizar_expr(self.bx_input.text()), locals=locals_dict))
            ay = float(sp.sympify(normalizar_expr(self.ay_input.text()), locals=locals_dict))
            by = float(sp.sympify(normalizar_expr(self.by_input.text()), locals=locals_dict))
            n = int(sp.sympify(normalizar_expr(self.n_input.text()), locals=locals_dict))
            semilla = int(sp.sympify(normalizar_expr(self.seed_input.text()), locals=locals_dict))
            conf = float(self.conf_input.text())

            (integral, (ic_inf, ic_sup),
             px, py, pz, desvio, error_estandar) = montecarlo_integral_doble(f, ax, bx, ay, by, n, semilla, conf)

            # --- Gráfico 3D ---
            for ax in self.figure.axes[:]:
                if ax is not self.ax:
                    self.figure.delaxes(ax)

            self.ax.clear()

            self.ax.set_position([0.1, 0.1, 0.7, 0.8])
            sc = self.ax.scatter(px, py, pz, c=pz, cmap="viridis", s=10, alpha=0.6)

            cax = self.figure.add_axes([0.8, 0.1, 0.03, 0.8])
            
            cbar = self.figure.colorbar(sc, cax=cax)
            cbar.set_label("f(x,y)")

            self.ax.yaxis.labelpad = 10
            self.ax.zaxis.labelpad = 12

            self.ax.set_title(f"Monte Carlo Doble (N={n}, IC {int(conf*100)}%)")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("f(x,y)")

            texto = (
                f"Integral ≈ {integral:.6f}\n"
                f"IC {int(conf*100)}%: [{ic_inf:.6f}, {ic_sup:.6f}]\n"
                f"N = {n}\n"
                f"Semilla = {semilla}\n"
                f"Desvío estándar = {desvio:.6f}\n"
                f"Error estándar = {error_estandar:.6f}"
            )
            self.figure.text(
                0.02, 0.9, texto,
                ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            self.canvas.draw()

            # --- Salida 
            self.output.clear()
            self.output.append("📊 Resultados del método Monte Carlo Doble:\n")
            self.output.append(f"Integral ≈ {integral}")
            self.output.append(f"IC {conf*100}%: [{ic_inf}, {ic_sup}]")
            self.output.append(f"N = {n}")
            self.output.append(f"Semilla = {semilla}")
            self.output.append(f"Desvío estándar = {desvio}")
            self.output.append(f"Error estándar = {error_estandar}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

# --- 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MonteCarloDobleWindow()
    window.show()
    sys.exit(app.exec())

