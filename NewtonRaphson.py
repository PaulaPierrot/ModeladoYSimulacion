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

# imports del teclado y otras cosas utiles.
from math_keyboard import MathKeyboard
from func_utils import normalizar_expr, crear_funcion_con_lhopital

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

# --- Método de Newton–Raphson ---
def newton_raphson(f, df, x0, tol, max_iter=100):
    pasos = []
    x = x0

    for n in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            raise ZeroDivisionError(f"Derivada nula en x={x}, no se puede continuar.")

        nuevo_x = x - fx / dfx
        error = abs(nuevo_x - x)
        pasos.append((n, x, fx, dfx, error))

        if error < tol:
            return nuevo_x, pasos
        x = nuevo_x

    raise ValueError("No converge en las iteraciones máximas")


class NewtonRaphsonWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Método de Newton–Raphson")
        self.setGeometry(100, 100, 1000, 700)
        self.active_input = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)

        self.f_input = QLineEdit("e^(x) - 3*x")
        self.df_input = QLineEdit("exp(x) - 3")
        self.x0_input = QLineEdit("0.5")
        self.tol_input = QLineEdit("0.001")

        for entrada in [self.f_input, self.df_input, self.x0_input, self.tol_input]:
            entrada.installEventFilter(self)

        form_layout.addWidget(QLabel("f(x):"))
        form_layout.addWidget(self.f_input)
        form_layout.addWidget(QLabel("f'(x):"))
        form_layout.addWidget(self.df_input)
        form_layout.addWidget(QLabel("x₀:"))
        form_layout.addWidget(self.x0_input)
        form_layout.addWidget(QLabel("Tolerancia:"))
        form_layout.addWidget(self.tol_input)

        self.keyboard = MathKeyboard(self.insertar_texto)
        layout.addWidget(self.keyboard)

        self.calc_btn = QPushButton("Calcular raíz (Newton–Raphson)")
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
            f_str = self.f_input.text()
            df_str = self.df_input.text()

            f = crear_funcion_con_lhopital(f_str)
            df = crear_funcion_con_lhopital(df_str)

            x0 = float(sp.sympify(normalizar_expr(self.x0_input.text()), locals={"pi": sp.pi, "E": sp.E}))
            tol = float(sp.sympify(normalizar_expr(self.tol_input.text()), locals={"pi": sp.pi, "E": sp.E}))

            raiz, pasos = newton_raphson(f, df, x0, tol)

            self.ax.clear()
            X = np.linspace(x0 - 3, x0 + 3, 400)
            Y = [f(val) for val in X]

            self.ax.plot(X, Y, "b", label="f(x)")
            self.ax.axhline(0, color="black", linewidth=0.8)
            self.ax.axvline(raiz, color="green", linestyle="--", label=f"Raíz ≈ {raiz}")
            self.ax.set_title("Método de Newton–Raphson")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

            self.output.clear()
            self.output.append(f"Raíz aproximada = {raiz}")
            self.output.append("Iteraciones:")
            for i, xi, fxi, dfxi, error in pasos:
                self.output.append(
                    f"Iter {i}: x={xi}, f(x)={fxi}, f'(x)={dfxi}, error={error}"
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# --- 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NewtonRaphsonWindow()
    window.show()
    sys.exit(app.exec())
