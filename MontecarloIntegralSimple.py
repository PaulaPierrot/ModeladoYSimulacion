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

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib

from func_utils import normalizar_expr, crear_funcion_con_lhopital
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


def montecarlo_integral(f, a, b, n, semilla, conf):
    random.seed(semilla)
    debajo_x, debajo_y = [], []
    arriba_x, arriba_y = [], []

    # calcular valor máximo de la función en el intervalo
    X = np.linspace(a, b, 500)
    Y = [f(val) for val in X]
    ymax = max(Y)

    # Monte Carlo
    valores_funcion = []
    for _ in range(n):
        x = random.uniform(a, b)
        y = random.uniform(0, ymax)
        fx = f(x)
        valores_funcion.append(fx)

        if y <= fx:
            debajo_x.append(x)
            debajo_y.append(y)
        else:
            arriba_x.append(x)
            arriba_y.append(y)

    # Integral estimada
    promedio = sum(valores_funcion) / len(valores_funcion)
    integral_estimada = (b - a) * promedio

    # Desvío y error estándar
    varianza = sum((v - promedio) ** 2 for v in valores_funcion) / (n - 1)
    desvio = math.sqrt(varianza)
    error_estandar = (b - a) * (desvio / math.sqrt(n))

    # Intervalo de confianza
    z = norm.ppf(1 - (1 - conf) / 2)
    z=round(z,3)
    intervalo_inf = integral_estimada - z * error_estandar
    intervalo_sup = integral_estimada + z * error_estandar

    return (integral_estimada, (intervalo_inf, intervalo_sup),
            debajo_x, debajo_y, arriba_x, arriba_y,
            X, Y, desvio, error_estandar)


class MonteCarloWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integración Numérica - Monte Carlo")
        self.setGeometry(100, 100, 1200, 700)
        self.active_input = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)

        self.fx_input = QLineEdit("sin(x)")
        self.a_input = QLineEdit("0")
        self.b_input = QLineEdit("pi")
        self.n_input = QLineEdit("10000")
        self.seed_input = QLineEdit("42")
        self.conf_input = QLineEdit("0.95")

        for entrada in [self.fx_input, self.a_input, self.b_input, self.n_input, self.seed_input, self.conf_input]:
            entrada.installEventFilter(self)

        form_layout.addWidget(QLabel("f(x):"))
        form_layout.addWidget(self.fx_input)
        form_layout.addWidget(QLabel("a:"))
        form_layout.addWidget(self.a_input)
        form_layout.addWidget(QLabel("b:"))
        form_layout.addWidget(self.b_input)
        form_layout.addWidget(QLabel("N:"))
        form_layout.addWidget(self.n_input)
        form_layout.addWidget(QLabel("Semilla:"))
        form_layout.addWidget(self.seed_input)
        form_layout.addWidget(QLabel("Confianza:"))
        form_layout.addWidget(self.conf_input)

        self.keyboard = MathKeyboard(self.insertar_texto)
        layout.addWidget(self.keyboard)

        self.calc_btn = QPushButton("Calcular")
        style_primary_button(self.calc_btn) 
        self.calc_btn.clicked.connect(self.calcular)
        layout.addWidget(self.calc_btn)

        
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
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
            f = crear_funcion_con_lhopital(expr_fx)

            locals_dict = {"pi": sp.pi, "π": sp.pi, "e": sp.E}
            a = float(sp.sympify(normalizar_expr(self.a_input.text()), locals=locals_dict))
            b = float(sp.sympify(normalizar_expr(self.b_input.text()), locals=locals_dict))
            n = int(sp.sympify(normalizar_expr(self.n_input.text()), locals=locals_dict))
            semilla = int(sp.sympify(normalizar_expr(self.seed_input.text()), locals=locals_dict))
            conf = float(self.conf_input.text())

            (integral, (ic_inf, ic_sup),
             dx, dy, ax_, ay_,
             X, Y, desvio, error_estandar) = montecarlo_integral(f, a, b, n, semilla, conf)

            # --- gráfico ---
            self.ax.clear()
            self.ax.plot(X, Y, color="blue", label="f(x)")
            self.ax.scatter(dx, dy, color="green", s=10, alpha=0.5, label="Puntos debajo")
            self.ax.scatter(ax_, ay_, color="red", s=10, alpha=0.5, label="Puntos arriba")
            self.ax.set_title(f"Monte Carlo (N={n}, IC {conf*100}%)")

            texto = (
                f"Integral ≈ {integral}\n"
                f"IC {conf*100}%: [{ic_inf}, {ic_sup}]\n"
                f"N = {n}\n"
                f"Semilla = {semilla}\n"
                f"Desvío estándar = {desvio}\n"
                f"Error estándar = {error_estandar}"
            )
            self.ax.text(
                0.05, 0.95, texto,
                transform=self.ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
            )

            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

            # --- Salida 
            self.output.clear()
            self.output.append("📊 Resultados del método Monte Carlo:\n")
            self.output.append(f"Integral ≈ {integral}")
            self.output.append(f"IC {conf*100}%: [{ic_inf}, {ic_sup}]")
            self.output.append(f"N = {n}")
            self.output.append(f"Semilla = {semilla}")
            self.output.append(f"Desvío estándar = {desvio}")
            self.output.append(f"Error estándar = {error_estandar}")
            
            # --- Paso a paso ---
            # --- Salida principal ---
            self.output.clear()
            self.output.append("📊 Resultados del método Monte Carlo:\n")
            self.output.append(f"Integral ≈ {integral}")
            self.output.append(f"IC {conf*100}%: [{ic_inf}, {ic_sup}]")
            self.output.append(f"N = {n}")
            self.output.append(f"Semilla = {semilla}")
            self.output.append(f"Desvío estándar = {desvio}")
            self.output.append(f"Error estándar = {error_estandar}")

            # --- Paso a paso con fórmulas ---
            self.output.append("\n📝 Paso a paso del cálculo:")

            # 1) ymax
            self.output.append("1) Se calcula el valor máximo de f(x) en [a,b]:")
            self.output.append("   Fórmula:  y_max = max{ f(x),  x ∈ [a,b] }")
            self.output.append(f"   Resultado: y_max = {max(Y)}")

            # 2) Generación de puntos
            self.output.append("2) Se generan N puntos aleatorios uniformes en el rectángulo:")
            self.output.append("   Región: [a,b] × [0,y_max]")
            self.output.append(f"   Valores: [{a},{b}] × [0,{max(Y)}] con N={n}")

            # 3) Clasificación
            self.output.append("3) Clasificación de puntos:")
            self.output.append("   Condición: si y ≤ f(x) → debajo, en caso contrario → arriba")
            self.output.append(f"   Resultado: debajo = {len(dx)}, arriba = {len(ax_)}")

            # 4) Estimación de la integral
            self.output.append("4) Integral estimada:")
            self.output.append("   Fórmula:  I ≈ (b-a) * (1/N) Σ f(x_i)")
            self.output.append(f"   Sustitución: I ≈ ({b}-{a}) * ({promedio})")
            self.output.append(f"   Resultado: I ≈ {integral}")

            # 5) Varianza y desvío estándar
            self.output.append("5) Varianza y desvío:")
            self.output.append("   Fórmula:  s² = (1/(N-1)) Σ (f(x_i) - promedio)²")
            self.output.append(f"   Resultado: varianza ≈ {desvio**2}, desvío ≈ {desvio}")

            # 6) Error estándar
            self.output.append("6) Error estándar:")
            self.output.append("   Fórmula:  E = (b-a) * (s / √N)")
            self.output.append(f"   Sustitución: E = ({b}-{a}) * ({desvio}/√{n})")
            self.output.append(f"   Resultado: {error_estandar}")

            # 7) Intervalo de confianza
            z = norm.ppf(1 - (1-conf)/2)
            self.output.append("7) Intervalo de confianza:")
            self.output.append("   Fórmula: IC = I ± z * E")
            self.output.append(f"   Sustitución: IC = {integral} ± {z} * {error_estandar}")
            self.output.append(f"   Resultado: [{ic_inf}, {ic_sup}]")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# --- 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MonteCarloWindow()
    window.show()
    sys.exit(app.exec())
