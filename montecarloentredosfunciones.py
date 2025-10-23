from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QComboBox, QGraphicsDropShadowEffect
)
import sys, random, math
import sympy as sp
import matplotlib
matplotlib.use("QtAgg")   # 👈 importante: backend correcto para PyQt6
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.stats import norm

# --- estilo de botón ---
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

# --- Monte Carlo rechazo (x o y) ---
def montecarlo_rechazo(f, g, a, b, n, semilla, conf, modo="x"):
    random.seed(semilla)
    dentro, fuera = 0, 0
    puntos_x, puntos_y, colores = [], [], []
    valores = []

    if modo == "x":
        ymax = 1.0
        for _ in range(n):
            x = random.uniform(a, b)
            y = random.uniform(0, ymax)
            if f(x) <= y <= g(x):
                dentro += 1
                colores.append("green")
                valores.append(1)
            else:
                fuera += 1
                colores.append("red")
                valores.append(0)
            puntos_x.append(x)
            puntos_y.append(y)

        area_rect = (b - a) * ymax
        area_estimada = (dentro / n) * area_rect

    else:  # modo "y"
        xmax = 1.0
        for _ in range(n):
            y = random.uniform(a, b)
            x = random.uniform(0, xmax)
            if f(y) <= x <= g(y):
                dentro += 1
                colores.append("green")
                valores.append(1)
            else:
                fuera += 1
                colores.append("red")
                valores.append(0)
            puntos_x.append(x)
            puntos_y.append(y)

        area_rect = (b - a) * xmax
        area_estimada = (dentro / n) * area_rect

    # --- estadísticos ---
    promedio = sum(valores) / n
    varianza = sum((v - promedio) ** 2 for v in valores) / (n - 1)
    desvio = math.sqrt(varianza)
    error_estandar = area_rect * (desvio / math.sqrt(n))

    z = norm.ppf(1 - (1 - conf) / 2)
    intervalo_inf = area_estimada - z * error_estandar
    intervalo_sup = area_estimada + z * error_estandar

    return (area_estimada, puntos_x, puntos_y, colores,
            area_rect, dentro, fuera, varianza, desvio,
            error_estandar, (intervalo_inf, intervalo_sup))

# --- Ventana principal ---
class MonteCarloRechazoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monte Carlo - Rechazo (x/y)")
        self.setGeometry(100, 100, 1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # inputs
        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)

        self.fx_input = QLineEdit("x**2")
        self.gx_input = QLineEdit("sqrt(x)")
        self.a_input = QLineEdit("0")
        self.b_input = QLineEdit("1")
        self.n_input = QLineEdit("5000")
        self.seed_input = QLineEdit("42")
        self.conf_input = QLineEdit("0.95")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["x", "y"])  # variable principal

        form_layout.addWidget(QLabel("f(x) o f(y):"))
        form_layout.addWidget(self.fx_input)
        form_layout.addWidget(QLabel("g(x) o g(y):"))
        form_layout.addWidget(self.gx_input)
        form_layout.addWidget(QLabel("Intervalo [a,b]:"))
        form_layout.addWidget(self.a_input)
        form_layout.addWidget(self.b_input)
        form_layout.addWidget(QLabel("N:"))
        form_layout.addWidget(self.n_input)
        form_layout.addWidget(QLabel("Semilla:"))
        form_layout.addWidget(self.seed_input)
        form_layout.addWidget(QLabel("Confianza:"))
        form_layout.addWidget(self.conf_input)
        form_layout.addWidget(QLabel("Variable:"))
        form_layout.addWidget(self.mode_combo)

        # botón
        self.calc_btn = QPushButton("Calcular área")
        style_primary_button(self.calc_btn)
        self.calc_btn.clicked.connect(self.calcular)
        layout.addWidget(self.calc_btn)

        # gráfico
        self.figure, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # salida
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    def calcular(self):
        try:
            expr_f = sp.sympify(self.fx_input.text())
            expr_g = sp.sympify(self.gx_input.text())
            var = sp.symbols("x" if self.mode_combo.currentText() == "x" else "y")
            f = sp.lambdify(var, expr_f, "math")
            g = sp.lambdify(var, expr_g, "math")

            a = float(self.a_input.text())
            b = float(self.b_input.text())
            n = int(self.n_input.text())
            seed = int(self.seed_input.text())
            conf = float(self.conf_input.text())
            modo = self.mode_combo.currentText()

            (area_estimada, px, py, colores, area_rect,
             dentro, fuera, varianza, desvio,
             error_estandar, (ic_inf, ic_sup)) = montecarlo_rechazo(
                f, g, a, b, n, seed, conf, modo
            )

            # --- gráfico
            self.ax.clear()
            self.ax.scatter(px, py, c=colores, s=10, alpha=0.5)
            self.ax.set_title(f"Monte Carlo Rechazo (modo {modo})")
            self.canvas.draw()

            # --- salida con paso a paso
            self.output.clear()
            self.output.append("📊 Resultados del método Monte Carlo por rechazo:\n")
            self.output.append(f"Área del rectángulo = {area_rect}")
            self.output.append(f"Puntos dentro = {dentro}, Puntos fuera = {fuera}")
            self.output.append(f"Área estimada ≈ {area_estimada}")
            self.output.append(f"Varianza ≈ {varianza}")
            self.output.append(f"Desvío estándar ≈ {desvio}")
            self.output.append(f"Error estándar ≈ {error_estandar}")
            self.output.append(f"IC {int(conf*100)}%: [{ic_inf}, {ic_sup}]\n")

            self.output.append("📝 Paso a paso del cálculo:")
            self.output.append(f"1) Definición del rectángulo de muestreo según modo {modo}")
            self.output.append(f"   Intervalo: [{a},{b}] con N = {n}")
            self.output.append("2) Generación de N puntos aleatorios uniformes en el rectángulo")
            self.output.append("3) Clasificación: dentro si f(var) <= otra_var <= g(var)")
            self.output.append("   - Puntos dentro = verdes")
            self.output.append("   - Puntos fuera = rojos")
            self.output.append("4) Estimación del área:")
            self.output.append("   Fórmula: Área ≈ (puntos dentro / N) * Área rectángulo")
            self.output.append(f"   Sustitución: ({dentro}/{n}) * {area_rect}")
            self.output.append(f"   Resultado: {area_estimada}")
            self.output.append("5) Varianza y desvío estándar:")
            self.output.append("   Fórmula: s² = (1/(N-1)) Σ (Xi - promedio)²")
            self.output.append(f"   Resultado: varianza ≈ {varianza}, desvío ≈ {desvio}")
            self.output.append("6) Error estándar:")
            self.output.append("   Fórmula: EE = Área_rect * (s / √N)")
            self.output.append(f"   Resultado: {error_estandar}")
            self.output.append("7) Intervalo de confianza:")
            self.output.append("   Fórmula: IC = I ± z * EE")
            self.output.append(f"   Resultado: [{ic_inf}, {ic_sup}]")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

# ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MonteCarloRechazoWindow()
    window.show()
    sys.exit(app.exec())
