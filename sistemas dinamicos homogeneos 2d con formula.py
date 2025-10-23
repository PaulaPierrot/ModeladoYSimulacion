from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox,
    QGraphicsDropShadowEffect, QComboBox
)
import sys
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib

matplotlib.use("QtAgg")


# --- estilo del botón ---
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
    """)
    shadow = QGraphicsDropShadowEffect(btn)
    shadow.setBlurRadius(22)
    shadow.setOffset(0, 2)
    shadow.setColor(QColor(0, 0, 0, 70))
    btn.setGraphicsEffect(shadow)


class FaseLineal2DWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diagrama de Fases - Sistemas Lineales 2D (Homogéneos)")
        self.setGeometry(100, 100, 1200, 820)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- inputs ---
        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)

        self.xdot_input = QLineEdit("x + y")
        self.ydot_input = QLineEdit("2*x - y")
        self.range_x_input = QLineEdit("-5,5")
        self.range_y_input = QLineEdit("-5,5")
        self.tmax_input = QLineEdit("10")
        self.ntraj_input = QLineEdit("14")

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Personalizado",
            "Nodo estable (diag. negativa)",
            "Nodo inestable (diag. positiva)",
            "Silla",
            "Foco estable",
            "Foco inestable",
            "Centro (rotación pura)",
            "Degenerado (Jordan λ=0)"
        ])
        self.preset_combo.currentIndexChanged.connect(self._aplicar_preset)

        form_layout.addWidget(QLabel("ẋ ="))
        form_layout.addWidget(self.xdot_input)
        form_layout.addWidget(QLabel("ẏ ="))
        form_layout.addWidget(self.ydot_input)
        form_layout.addWidget(QLabel("Rango x:"))
        form_layout.addWidget(self.range_x_input)
        form_layout.addWidget(QLabel("Rango y:"))
        form_layout.addWidget(self.range_y_input)
        form_layout.addWidget(QLabel("t máx:"))
        form_layout.addWidget(self.tmax_input)
        form_layout.addWidget(QLabel("#trayec:"))
        form_layout.addWidget(self.ntraj_input)
        form_layout.addWidget(QLabel("Preset:"))
        form_layout.addWidget(self.preset_combo)

        # --- botón ---
        self.calc_btn = QPushButton("Analizar sistema")
        style_primary_button(self.calc_btn)
        self.calc_btn.clicked.connect(self.analizar)
        layout.addWidget(self.calc_btn)

        # --- gráfico ---
        self.figure, self.ax = plt.subplots(1, 1, figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # --- salida texto ---
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    # ================== helpers ==================
    def _aplicar_preset(self):
        presets = {
            1: ("-x", "-2*y"),
            2: ("x", "2*y"),
            3: ("x", "-y"),
            4: ("-0.5*x - 2*y", "2*x - 0.5*y"),
            5: ("0.5*x - 2*y", "2*x + 0.5*y"),
            6: ("-y", "x"),
            7: ("x + y", "-x - y"),
        }
        i = self.preset_combo.currentIndex()
        if i in presets:
            xdot, ydot = presets[i]
            self.xdot_input.setText(xdot)
            self.ydot_input.setText(ydot)

    def _lambdas(self, fx, gy):
        return sp.lambdify((self.x, self.y), fx, "numpy"), sp.lambdify((self.x, self.y), gy, "numpy")

    def _clasificar(self, A):
        tau = float(A.trace())
        det = float(A.det())
        disc = tau * tau - 4 * det

        if det < 0:
            tipo = "Silla (saddle)"
        elif det > 0 and disc > 0:
            tipo = "Nodo estable (τ<0)" if tau < 0 else "Nodo inestable (τ>0)"
        elif det > 0 and disc < 0:
            tipo = "Foco estable (τ<0)" if tau < 0 else ("Foco inestable (τ>0)" if tau > 0 else "Centro")
        elif det == 0 and tau == 0:
            tipo = "Degenerado nulo"
        else:
            tipo = "Nodo degenerado (λ doble)"
        return tipo, tau, det, disc

    # ================== análisis ==================
    def analizar(self):
        try:
            self.x, self.y = sp.symbols("x y")
            fx = sp.sympify(self.xdot_input.text(), locals={"x": self.x, "y": self.y})
            gy = sp.sympify(self.ydot_input.text(), locals={"x": self.x, "y": self.y})

            x_min, x_max = [float(v) for v in self.range_x_input.text().split(",")]
            y_min, y_max = [float(v) for v in self.range_y_input.text().split(",")]

            A = sp.Matrix([[sp.diff(fx, self.x), sp.diff(fx, self.y)],
                           [sp.diff(gy, self.x), sp.diff(gy, self.y)]])
            A_simpl = sp.simplify(A)
            eig_data = A_simpl.eigenvects()

            tipo, tau, det, disc = self._clasificar(A_simpl)
            fnum, gnum = self._lambdas(fx, gy)

            # --- gráfico ---
            self.ax.clear()
            Xg, Yg = np.meshgrid(np.linspace(x_min, x_max, 25), np.linspace(y_min, y_max, 25))
            U = fnum(Xg, Yg)
            V = gnum(Xg, Yg)
            self.ax.streamplot(Xg, Yg, U, V, density=1.2, linewidth=1)
            self.ax.plot(0, 0, "ko", ms=5)
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.ax.grid(True, linestyle='--', alpha=0.5)
            self.ax.set_title("Diagrama de Fases")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.canvas.draw()

            # --- salida analítica ---
            self.output.clear()
            self.output.append("Sistema:")
            self.output.append(f"  ẋ = {sp.simplify(fx)}")
            self.output.append(f"  ẏ = {sp.simplify(gy)}\n")

            self.output.append("Matriz A =")
            self.output.append(str(A_simpl))
            self.output.append("")

            self.output.append(f"τ = {tau:.3f}, det = {det:.3f}, D = {disc:.3f}")
            self.output.append(f"Clasificación: {tipo}\n")

            self.output.append("Autovalores y autovectores:")
            for ev in eig_data:
                self.output.append(f"  λ = {sp.N(ev[0])}, mult. {ev[1]}")
                for j, v in enumerate(ev[2], start=1):
                    self.output.append(f"     v{j} = {sp.N(sp.Matrix(v))}")
            self.output.append("")

            # --- solución general ---
            c1, c2, t = sp.symbols("c1 c2 t")

            # caso 1: autovalor doble
            if len(eig_data) == 1 and eig_data[0][1] == 2:
                lam, _, vecs = eig_data[0]
                if len(vecs) == 2:
                    # diagonalizable (nodo estrella)
                    X_t = sp.exp(lam * t) * (c1 * sp.Matrix(vecs[0]) + c2 * sp.Matrix(vecs[1]))
                else:
                    # Jordan
                    v = sp.Matrix(vecs[0])
                    M = A_simpl - lam * sp.eye(2)
                    w = sp.Matrix(sp.symbols("w1 w2"))
                    sol = sp.linsolve((M, v))
                    wv = list(sol)[0] if sol else (0, 0)
                    w = sp.Matrix(wv)
                    X_t = sp.exp(lam * t) * (c1 * v + c2 * (t * v + w))
            # caso 2: complejos
            elif any(sp.im(ev[0]) != 0 for ev in eig_data):
                lam = eig_data[0][0]
                v = eig_data[0][2][0]
                α, β = sp.re(lam), sp.im(lam)
                a, b = sp.re(v), sp.im(v)
                X_t = sp.exp(α * t) * (
                    c1 * (a * sp.cos(β * t) - b * sp.sin(β * t)) +
                    c2 * (a * sp.sin(β * t) + b * sp.cos(β * t))
                )
            # caso 3: reales distintos
            else:
                (λ1, _, v1), (λ2, _, v2) = eig_data[0], eig_data[1]
                X_t = c1 * sp.exp(λ1 * t) * sp.Matrix(v1[0]) + c2 * sp.exp(λ2 * t) * sp.Matrix(v2[0])

            self.output.append("Solución general:")
            self.output.append(f"  X(t) = {sp.simplify(X_t)}\n")

            x_t, y_t = sp.simplify(X_t[0]), sp.simplify(X_t[1])
            self.output.append("Componentes:")
            self.output.append(f"  x(t) = {x_t}")
            self.output.append(f"  y(t) = {y_t}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaseLineal2DWindow()
    window.show()
    sys.exit(app.exec())
