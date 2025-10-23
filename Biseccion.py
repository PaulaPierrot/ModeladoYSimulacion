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


# --- metodo de biseccion 
def biseccion(f, a, b, tol, max_iter=100):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos.")

    iteraciones = []
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        iteraciones.append((i+1, a, b, c, fc))

        if abs(fc) < tol or abs(b-a)/2 < tol:
            return c, iteraciones

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    raise ValueError("No converge en las iteraciones máximas")


class BiseccionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bisección - Búsqueda binaria de raíces")
        self.setGeometry(100, 100, 1000, 700)
        self.active_input = None 

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)
        
        #Acá le puse esto por defecto para que sea más facil testear
        self.func_input = QLineEdit("sqrt(x) - cos(x)")
        self.a_input = QLineEdit("0")
        self.b_input = QLineEdit("1")
        self.tol_input = QLineEdit("0.001")

        for entrada in [self.func_input, self.a_input, self.b_input, self.tol_input]:
            entrada.installEventFilter(self)

        form_layout.addWidget(QLabel("f(x):"))
        form_layout.addWidget(self.func_input)
        form_layout.addWidget(QLabel("a:"))
        form_layout.addWidget(self.a_input)
        form_layout.addWidget(QLabel("b:"))
        form_layout.addWidget(self.b_input)
        form_layout.addWidget(QLabel("Tolerancia:"))
        form_layout.addWidget(self.tol_input)

        # --- teclado matematico importado del otro activo 
        self.keyboard = MathKeyboard(self.insertar_texto)
        layout.addWidget(self.keyboard)

        # --- boton para calcular
        self.calc_btn = QPushButton("Calcular raíz (Bisección)")
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
        """detecta qué input estamos usando"""
        if event.type() == event.Type.FocusIn:
            self.active_input = obj
        return super().eventFilter(obj, event)

    def insertar_texto(self, texto):
        """inserta texto en el input activo"""
        if not self.active_input:
            return
        cursor = self.active_input.cursorPosition()
        actual = self.active_input.text()
        nuevo = actual[:cursor] + texto + actual[cursor:]
        self.active_input.setText(nuevo)
        self.active_input.setCursorPosition(cursor + len(texto))

    def calcular(self):
        try:
            expr_str = self.func_input.text()
            f = crear_funcion_con_lhopital(expr_str)
            
            a = float(sp.sympify(normalizar_expr(self.a_input.text()), locals={"pi": sp.pi, "E": sp.E}))
            b = float(sp.sympify(normalizar_expr(self.b_input.text()), locals={"pi": sp.pi, "E": sp.E}))
            tol = float(sp.sympify(normalizar_expr(self.tol_input.text()), locals={"pi": sp.pi, "E": sp.E}))

            raiz, pasos = biseccion(f, a, b, tol)

            # --- grafico ---
            self.ax.clear()
            X = np.linspace(a, b, 400)
            Y = [f(x) for x in X]
            self.ax.axhline(0, color="black")
            self.ax.plot(X, Y, "b", label="f(x)")
            self.ax.axvline(raiz, color="red", linestyle="--", label=f"Raíz ≈ {raiz}")
            self.ax.set_title("Método de Bisección")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

            self.output.clear()
            self.output.append(f"Raíz aproximada = {raiz}")
            self.output.append("Iteraciones:")
            for i, ai, bi, ci, fci in pasos:
                self.output.append(f"Iteración {i}: a={ai}, b={bi}, c={ci}, f(c)={fci}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# --- main 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BiseccionWindow()
    window.show()
    sys.exit(app.exec())
