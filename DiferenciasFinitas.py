from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import sympy as sp
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


# --- Aproximaciones por diferencias ---
def diferencias_finitas(f, x0, h):
    f_x0 = f(x0)
    f_x0_h = f(x0 + h)
    f_x0_mh = f(x0 - h)

    progresiva = (f_x0_h - f_x0) / h
    regresiva = (f_x0 - f_x0_mh) / h
    central = (f_x0_h - f_x0_mh) / (2 * h)

    pasos = []
    pasos.append(f"Progresiva: (f(x₀+h) - f(x₀)) / h = ({f_x0_h} - {f_x0}) / {h} = {progresiva}")
    pasos.append(f"Regresiva: (f(x₀) - f(x₀-h)) / h = ({f_x0} - {f_x0_mh}) / {h} = {regresiva}")
    pasos.append(f"Central: (f(x₀+h) - f(x₀-h)) / (2h) = ({f_x0_h} - {f_x0_mh}) / (2*{h}) = {central}")

    return progresiva, regresiva, central, pasos


class DiferenciasFinitasWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diferencias Finitas (Progresiva, Regresiva, Central)")
        self.setGeometry(100, 100, 1200, 700)
        self.active_input = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)

        self.fx_input = QLineEdit("sin(x)")
        self.x0_input = QLineEdit("0.5")
        self.h_input = QLineEdit("0.1")

        for entrada in [self.fx_input, self.x0_input, self.h_input]:
            entrada.installEventFilter(self)

        form_layout.addWidget(QLabel("f(x):"))
        form_layout.addWidget(self.fx_input)
        form_layout.addWidget(QLabel("x₀:"))
        form_layout.addWidget(self.x0_input)
        form_layout.addWidget(QLabel("h:"))
        form_layout.addWidget(self.h_input)

        self.keyboard = MathKeyboard(self.insertar_texto)
        layout.addWidget(self.keyboard)

        self.calc_btn = QPushButton("Calcular derivadas aproximadas")
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

            x0 = float(sp.sympify(normalizar_expr(self.x0_input.text()), locals={"pi": sp.pi, "E": sp.E}))
            h = float(sp.sympify(normalizar_expr(self.h_input.text()), locals={"pi": sp.pi, "E": sp.E}))

            progresiva, regresiva, central, pasos = diferencias_finitas(f, x0, h)

            # --- gráfico ---
            self.ax.clear()
            X = np.linspace(x0 - 2, x0 + 2, 400)
            Y = [f(val) for val in X]
            self.ax.plot(X, Y, "b", label="f(x)")
            self.ax.plot(x0, f(x0), "ro", label=f"punto (x₀, f(x₀))")

            # rectas aproximadas
            for deriv, estilo, nombre in [
                (progresiva, "g--", "Tangente (progresiva)"),
                (regresiva, "m--", "Tangente (regresiva)"),
                (central, "c-", "Tangente (central)")
            ]:
                recta = lambda x: f(x0) + deriv * (x - x0)
                self.ax.plot(X, [recta(xx) for xx in X], estilo, label=nombre)

            # resultados en el gráfico
            texto_resultados = (
                f"Progresiva = {progresiva:.5f}\n"
                f"Regresiva = {regresiva:.5f}\n"
                f"Central = {central:.5f}"
            )
            self.ax.text(
                0.98, 0.95, texto_resultados,
                transform=self.ax.transAxes,
                fontsize=10, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
            )

            self.ax.set_title("Aproximación de la derivada con diferencias finitas")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

            # --- 
            self.output.clear()
            self.output.append("Procedimiento paso a paso:\n")
            for paso in pasos:
                self.output.append(paso)

            self.output.append("\n Resultados finales:")
            self.output.append(f" Progresiva: {progresiva}")
            self.output.append(f" Regresiva: {regresiva}")
            self.output.append(f" Central: {central}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# --- 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DiferenciasFinitasWindow()
    window.show()
    sys.exit(app.exec())
