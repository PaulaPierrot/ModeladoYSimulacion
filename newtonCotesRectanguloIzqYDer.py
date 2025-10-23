from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QLabel
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
import sys
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from func_utils import normalizar_expr, crear_funcion_con_lhopital
from math_keyboard import MathKeyboard

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtGui import QPixmap, QImage
import io
import matplotlib

matplotlib.use("QtAgg")


# --- Función para renderizar LaTeX en QLabel ---
def render_latex(latex_str):
    fig = plt.figure(figsize=(3, 0.7)) 
    fig.text(0.5, 0.5, latex_str, fontsize=11, ha='center', va='center')
    buf = io.BytesIO()
    plt.axis("off")
    plt.savefig(buf, format="png", dpi=70, bbox_inches="tight", pad_inches=0.05, transparent=False, facecolor="white") 
    plt.close(fig)
    buf.seek(0)
    image = QImage.fromData(buf.read())
    return QPixmap(image)


# --- Error teórico ---
def estimar_error(expr_str, a, b, h, metodo):
    x = sp.symbols('x')
    expr = sp.sympify(expr_str)
    if metodo in ["left", "right"]:
        deriv = sp.diff(expr, x, 1)   # primera derivada
        f_deriv = sp.lambdify(x, deriv, 'numpy')
        X = np.linspace(a, b, 500)
        max_deriv = max(abs(f_deriv(val)) for val in X)
        factor = 0.5 * (b - a) * h
        return factor * max_deriv if metodo == "left" else -factor * max_deriv

    elif metodo == "midpoint":
        deriv = sp.diff(expr, x, 2)   # segunda derivada
        f_deriv = sp.lambdify(x, deriv, 'numpy')
        X = np.linspace(a, b, 500)
        max_deriv = max(abs(f_deriv(val)) for val in X)
        return -((b - a) / 24) * (h ** 2) * max_deriv
    elif metodo == "trapezoidal":
        deriv = sp.diff(expr, x, 2)
        f_deriv = sp.lambdify(x, deriv, 'numpy')
        X = np.linspace(a, b, 500)
        max_deriv = max(abs(f_deriv(val)) for val in X)
        return -((b - a) / 12) * (h ** 2) * max_deriv
    elif metodo in ["simpson13", "simpson38"]:
        deriv = sp.diff(expr, x, 4)
        f_deriv = sp.lambdify(x, deriv, 'numpy')
        X = np.linspace(a, b, 500)
        max_deriv = max(abs(f_deriv(val)) for val in X)
        if metodo == "simpson13":
            return -((b - a) / 180) * (h ** 4) * max_deriv
        else:
            return -((3 * (b - a)) / 80) * (h ** 4) * max_deriv
    return None

class ArrowComboBox(QComboBox):
    def __init__(self, *args, arrow="⤵", box_w=28, **kwargs):
        super().__init__(*args, **kwargs)
        self._arrow_lbl = QLabel(arrow, self)
        self._arrow_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._arrow_lbl.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._arrow_lbl.setStyleSheet("color:#0f172a; font-weight:800;")
        self._box_w = box_w

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._arrow_lbl.setGeometry(self.width()-self._box_w, 0,
                                    self._box_w, self.height())
        
def style_primary_combo(cbo, *, radius=12, accent="#0ea5e9", height=34, box_w=28):
    cbo.setFixedHeight(height)
    cbo.setCursor(Qt.CursorShape.PointingHandCursor)
    cbo.setStyleSheet(f"""
        QComboBox {{
            background: #ffffff;
            color: #0f172a;
            border: 1px solid rgba(0,0,0,0.12);
            border-radius: {radius}px;
            padding: 0 {box_w+4}px 0 12px;   /* espacio para “⤵” */
            font-weight: 700;
        }}
        QComboBox:hover {{
            background: #f6f8fb;
            border-color: {accent};
        }}
        /* ocultamos la flecha nativa y dejamos el carril a la derecha */
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: {box_w}px;
            border-left: 1px solid rgba(0,0,0,0.06);
            border-top-right-radius: {radius}px;
            border-bottom-right-radius: {radius}px;
        }}
        QComboBox::down-arrow {{ image: none; }}
        /* popup */
        QComboBox QAbstractItemView {{
            background: #ffffff;
            color: #0f172a;
            border: 1px solid rgba(0,0,0,0.15);
            selection-background-color: #eef2f7;
            selection-color: #0f172a;
            outline: 0;
        }}
    """)
    shadow = QGraphicsDropShadowEffect(cbo)
    shadow.setBlurRadius(18)
    shadow.setOffset(0, 2)
    shadow.setColor(QColor(0, 0, 0, 70))
    cbo.setGraphicsEffect(shadow)


def style_primary_button(btn, *, radius=12, accent="#0ea5e9", height=34):
    btn.setFixedHeight(height)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: #ffffff;
            color: #0f172a;
            border: 1px solid rgba(0,0,0,0.12);
            border-radius: {radius}px;
            padding: 6px 12px;        /* más bajo */
            font-weight: 800;
        }}
        QPushButton:hover {{
            background: #f6f8fb;
            border-color: {accent};
        }}
        QPushButton:pressed {{
            background: #eef2f7;
            padding-top: 7px;
        }}
        QPushButton:disabled {{
            color: rgba(15,23,42,0.35);
            border-color: rgba(0,0,0,0.08);
            background: #fafafa;
        }}
    """)
    shadow = QGraphicsDropShadowEffect(btn)
    shadow.setBlurRadius(18)
    shadow.setOffset(0, 2)
    shadow.setColor(QColor(0, 0, 0, 70))
    btn.setGraphicsEffect(shadow)


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


# --- Métodos Newton-Cotes ---
def newton_cotes(f, a, b, n, metodo):
    h = (b - a) / n
    x_vals = [a + i * h for i in range(n + 1)]
    y_vals = [f(x) for x in x_vals]
    if metodo == "left":
        integral = h * sum(y_vals[:-1])
        return integral, x_vals[:-1], y_vals[:-1]

    elif metodo == "right":
        integral = h * sum(y_vals[1:])
        return integral, x_vals[1:], y_vals[1:]

    elif metodo == "midpoint":
        midpoints = [a + (i + 0.5) * h for i in range(n)]
        ys = [f(x) for x in midpoints]
        return h * sum(ys), midpoints, ys
    elif metodo == "trapezoidal":
        integral = h * (0.5 * y_vals[0] + sum(y_vals[1:-1]) + 0.5 * y_vals[-1])
        return integral, x_vals, y_vals
    elif metodo == "simpson13":
        if n % 2 != 0:
            raise ValueError("Simpson 1/3 requiere n par.")
        integral = h / 3 * (y_vals[0] + y_vals[-1] +
                            4 * sum(y_vals[1:-1:2]) +
                            2 * sum(y_vals[2:-2:2]))
        return integral, x_vals, y_vals
    elif metodo == "simpson38":
        if n % 3 != 0:
            raise ValueError("Simpson 3/8 requiere n múltiplo de 3.")
        suma_no_mult3 = sum(y_vals[i] for i in range(1, n) if i % 3 != 0)
        suma_mult3 = sum(y_vals[i] for i in range(3, n, 3))
        integral = (3 * h / 8) * (y_vals[0] + y_vals[-1] +
                                 3 * suma_no_mult3 + 2 * suma_mult3)
        return integral, x_vals, y_vals
    raise ValueError("Método no reconocido.")


class NewtonCotesWindow(QMainWindow):  
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integración Numérica - Newton Cotes")
        self.setGeometry(100, 100, 1200, 700)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)
        self.active_input = None
        self.func_input = QLineEdit("6+3*cos(x)")
        self.a_input = QLineEdit("0")
        self.b_input = QLineEdit("pi")
        self.n_input = QLineEdit("4")
        for entrada in [self.func_input, self.a_input, self.b_input, self.n_input]:
            entrada.installEventFilter(self)
        self.method_dropdown = QComboBox()
        self.method_dropdown = ArrowComboBox(arrow="↴", box_w=28)
        self.method_dropdown.addItems(["left", "right", "midpoint", "trapezoidal", "simpson13", "simpson38"])
        style_primary_combo(self.method_dropdown, height=34, radius=12)
        form_layout.addWidget(QLabel("f(x):"))
        form_layout.addWidget(self.func_input)
        form_layout.addWidget(QLabel("a:"))
        form_layout.addWidget(self.a_input)
        form_layout.addWidget(QLabel("b:"))
        form_layout.addWidget(self.b_input)
        form_layout.addWidget(QLabel("n:"))
        form_layout.addWidget(self.n_input)
        form_layout.addWidget(QLabel("Método:"))
        form_layout.addWidget(self.method_dropdown)
        self.keyboard = MathKeyboard(self.insertar_texto)
        layout.addWidget(self.keyboard)
        self.calc_btn = QPushButton("Calcular")
        style_primary_button(self.calc_btn) 
        self.calc_btn.clicked.connect(self.calcular)
        layout.addWidget(self.calc_btn)
        self.formulas_layout = QHBoxLayout()  
        layout.addLayout(self.formulas_layout)
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    def insertar_texto(self, texto):
        if not self.active_input:
            return
        cursor = self.active_input.cursorPosition()
        actual = self.active_input.text()
        nuevo = actual[:cursor] + texto + actual[cursor:]
        self.active_input.setText(nuevo)
        self.active_input.setCursorPosition(cursor + len(texto))

    def eventFilter(self, obj, event):
        if event.type() == event.Type.FocusIn:
            self.active_input = obj
        return super().eventFilter(obj, event)

    def mostrar_formulas(self, formulas_latex):
        while self.formulas_layout.count():
            item = self.formulas_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        for latex_str in formulas_latex:
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setPixmap(render_latex(latex_str))
            self.formulas_layout.addWidget(lbl)

    def calcular(self):
        try:
            expr_str = normalizar_expr(self.func_input.text())
            f = crear_funcion_con_lhopital(expr_str)
            locals_dict = {"pi": sp.pi, "π": sp.pi, "e": sp.E}
            a = float(sp.sympify(normalizar_expr(self.a_input.text()), locals=locals_dict))
            b = float(sp.sympify(normalizar_expr(self.b_input.text()), locals=locals_dict))
            n = int(sp.sympify(normalizar_expr(self.n_input.text()), locals=locals_dict))
            metodo = self.method_dropdown.currentText()
            if metodo == "simpson13" and n % 2 != 0:
                QMessageBox.critical(self, "Error", "Simpson 1/3 requiere n par.")
                return
            if metodo == "simpson38" and n % 3 != 0:
                QMessageBox.critical(self, "Error", "Simpson 3/8 requiere n múltiplo de 3.")
                return
            formulas = {
                "left": r"$\int_a^b f(x)\,dx \approx h \sum_{i=0}^{n-1} f(x_i)$",
                "right": r"$\int_a^b f(x)\,dx \approx h \sum_{i=1}^{n} f(x_i)$",
                "midpoint": r"$\int_a^b f(x)\,dx \approx h \sum f(x_i + \frac{h}{2})$",
                "trapezoidal": r"$\int_a^b f(x)\,dx \approx \frac{h}{2}[f(x_0)+2\sum_{i=1}^{n-1} f(x_i)+f(x_n)]$",
                "simpson13": r"$\int_a^b f(x)\,dx \approx \frac{h}{3}[f(x_0)+4\sum_{i=1,3...}^{n-1} f(x_i)+2\sum_{j=2,4...}^{n-2} f(x_j)+f(x_n)]$",
                "simpson38": r"$\int_a^b f(x)\,dx \approx \frac{3h}{8}[f(x_0)+3\sum_{i \notin 3k} f(x_i)+2\sum_{j=3,6...}^{n-3} f(x_j)+f(x_n)]$"
            }
            error_latex = {
                "left": r"$E_L = \frac{(b-a)}{2} h f'(\xi)$",
                "right": r"$E_R = -\frac{(b-a)}{2} h f'(\xi)$",
                "midpoint": r"$E_M = -\frac{(b-a)}{24} h^2 f''(\xi)$",
                "trapezoidal": r"$E_T = -\frac{(b-a)}{12} h^2 f''(\xi)$",
                "simpson13": r"$E_{S_{1/3}} = -\frac{(b-a)}{180} h^4 f^{(4)}(\xi)$",
                "simpson38": r"$E_{S_{3/8}} = -\frac{3(b-a)}{80} h^4 f^{(4)}(\xi)$"
            }
            integral, px, py = newton_cotes(f, a, b, n, metodo)
            h = (b - a) / n
            error_estimado = estimar_error(expr_str, a, b, h, metodo)
            formulas_latex = [
                formulas[metodo],
                error_latex[metodo],
                f"$\\text{{Integral}} \\approx {integral}$"
            ]
            if error_estimado is not None:
                formulas_latex.append(f"$\\text{{Error de truncamiento}} \\approx {error_estimado}$")
            self.mostrar_formulas(formulas_latex)
            self.ax.clear()
            X = np.linspace(a, b, 400)
            Y = [f(x) for x in X]
            self.ax.plot(X, Y, "b", label="f(x)")
            self.ax.scatter(px, py, c="red", label="Puntos")
            self.ax.set_title(f"Método {metodo} compuesto (n={n})")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()
            self.output.clear()
            self.output.append(f"Integral aproximada = {integral}")
            self.output.append("Pares (x,y):")
            for x, y in zip(px, py):
                self.output.append(f"({x}, {y})")
            # Mostrar también la fórmula con valores reemplazados (sin redondear)
            if metodo == "left":
                valores = " + ".join([f"{y}" for y in py])
                formula_texto = f"{h} * ({valores})"
            elif metodo == "right":
                valores = " + ".join([f"{y}" for y in py])
                formula_texto = f"{h} * ({valores})"
            elif metodo == "trapezoidal":
                valores_internos = " + ".join([f"{y}" for y in py[1:-1]])
                formula_texto = f"({h}/2) * [{py[0]} + 2({valores_internos}) + {py[-1]}]"
            elif metodo == "midpoint":
                valores_mid = " + ".join([f"{f(x)}" for x in px])
                formula_texto = f"{h} * ({valores_mid})"
            elif metodo == "simpson13":
                impares = " + ".join([f"{py[i]}" for i in range(1, n, 2)])
                pares = " + ".join([f"{py[i]}" for i in range(2, n-1, 2)])
                formula_texto = f"({h}/3) * [{py[0]} + 4({impares}) + 2({pares}) + {py[-1]}]"
            elif metodo == "simpson38":
                no_mult3 = " + ".join([f"{py[i]}" for i in range(1, n) if i % 3 != 0])
                mult3 = " + ".join([f"{py[i]}" for i in range(3, n, 3)])
                formula_texto = f"(3*{h}/8) * [{py[0]} + 3({no_mult3}) + 2({mult3}) + {py[-1]}]"
            else:
                formula_texto = ""

            if formula_texto:
                self.output.append("")
                self.output.append("Reemplazo en la fórmula:")
                self.output.append(formula_texto + f" = {integral}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NewtonCotesWindow() 
    window.show()
    sys.exit(app.exec())
