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

# --- estilos para botones (igual que en tu app) ---
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

        # La persona ingresa directamente las ecuaciones
        self.xdot_input = QLineEdit("x + y")
        self.ydot_input = QLineEdit("2*x - y")

        # Rango de ejes y parámetros de integración
        self.range_x_input = QLineEdit("-5,5")   # rango en eje X
        self.range_y_input = QLineEdit("-5,5")   # rango en eje Y
        self.tmax_input = QLineEdit("10")        # tiempo total de integración (solo hacia adelante)
        self.ntraj_input = QLineEdit("14")       # cantidad de trayectorias (semillas)

        # Presets prácticos
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Personalizado",
            "Nodo estable (diag. negativa)",
            "Nodo inestable (diag. positiva)",
            "Silla",
            "Foco estable",
            "Foco inestable",
            "Centro (rotación pura)"
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

        # --- gráfico único (fase) ---
        self.figure, self.ax = plt.subplots(1, 1, figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # --- salida de texto ---
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    # ================== helpers internos (integrados) ==================

    def _aplicar_preset(self):
        i = self.preset_combo.currentIndex()
        if i == 0:
            return
        presets = {
            1: ("-x", "-2*y"),                    # Nodo estable (τ<0, det>0, D>0)
            2: ("x", "2*y"),                      # Nodo inestable (τ>0, det>0, D>0)
            3: ("x", "-y"),                       # Silla (det<0)
            4: ("-0.5*x - 2*y", "2*x - 0.5*y"),   # Foco estable (τ<0, det>0, D<0)
            5: ("0.5*x - 2*y", "2*x + 0.5*y"),    # Foco inestable (τ>0, det>0, D<0)
            6: ("-y", "x"),                       # Centro (τ=0, det>0, D<0 con α=0)
        }
        xdot, ydot = presets.get(i, ("x + y", "2*x - y"))
        self.xdot_input.setText(xdot)
        self.ydot_input.setText(ydot)

    def _lambdas(self, fx, gy):
        return sp.lambdify((self.x, self.y), fx, "numpy"), sp.lambdify((self.x, self.y), gy, "numpy")

    def _rk4(self, f, g, x0, y0, t):
        dt = t[1] - t[0]
        X = np.empty_like(t)
        Y = np.empty_like(t)
        X[0], Y[0] = x0, y0
        for k in range(len(t) - 1):
            k1x = f(X[k], Y[k])
            k1y = g(X[k], Y[k])
            k2x = f(X[k] + 0.5*dt*k1x, Y[k] + 0.5*dt*k1y)
            k2y = g(X[k] + 0.5*dt*k1x, Y[k] + 0.5*dt*k1y)
            k3x = f(X[k] + 0.5*dt*k2x, Y[k] + 0.5*dt*k2y)
            k3y = g(X[k] + 0.5*dt*k2x, Y[k] + 0.5*dt*k2y)
            k4x = f(X[k] + dt*k3x, Y[k] + dt*k3y)
            k4y = g(X[k] + dt*k3x, Y[k] + dt*k3y)
            X[k+1] = X[k] + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
            Y[k+1] = Y[k] + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
        return X, Y

    def _trayectorias_forward(self, fnum, gnum, xr, yr, tmax, nseed=12):
        seeds_r = 0.6*min(xr[1]-xr[0], yr[1]-yr[0])
        ang = np.linspace(0, 2*np.pi, nseed, endpoint=False)
        seeds = np.c_[seeds_r*np.cos(ang), seeds_r*np.sin(ang)]
        t = np.linspace(0, float(tmax), 900)
        trajs = []
        for x0, y0 in seeds:
            Xf, Yf = self._rk4(lambda X,Y: fnum(X,Y), lambda X,Y: gnum(X,Y), x0, y0, t)
            trajs.append((Xf, Yf))
        return trajs

    def _clasificar(self, A):
        tau = float(A.trace())
        det = float(A.det())
        disc = tau*tau - 4.0*det

        if det < 0:
            tipo = "Silla (saddle): λ1·λ2 < 0"
        elif det > 0 and disc > 0:
            tipo = "Nodo estable (τ<0)" if tau < 0 else "Nodo inestable (τ>0)"
        elif det > 0 and disc < 0:
            tipo = "Foco/Espiral estable (τ<0)" if tau < 0 else ("Foco/Espiral inestable (τ>0)" if tau > 0 else "Centro (α=0)")
        else:
            if det == 0 and tau == 0:
                tipo = "Degenerado nulo (requiere análisis no lineal)"
            elif det == 0:
                tipo = "Semieje nulo (una λ=0)"
            else:
                tipo = "Nodo degenerado (λ doble; estrella o Jordan)"
        return tipo, tau, det, disc

    # ================== acción principal ==================

    def analizar(self):
        try:
            # símbolos y parseo
            self.x, self.y = sp.symbols("x y")
            fx_str = self.xdot_input.text().strip()
            gy_str = self.ydot_input.text().strip()

            fx = sp.sympify(fx_str, locals={"x": self.x, "y": self.y})
            gy = sp.sympify(gy_str, locals={"x": self.x, "y": self.y})

            # rangos y parámetros
            x_min, x_max = [float(v) for v in self.range_x_input.text().split(",")]
            y_min, y_max = [float(v) for v in self.range_y_input.text().split(",")]
            tmax = float(self.tmax_input.text())
            ntraj = int(self.ntraj_input.text())

            # Jacobiano constante (sistema lineal homogéneo)
            A = sp.Matrix([[sp.diff(fx, self.x), sp.diff(fx, self.y)],
                           [sp.diff(gy, self.x), sp.diff(gy, self.y)]])
            A_simpl = sp.simplify(A)

            # autovalores / autovectores (simbólico)
            eig_data = A_simpl.eigenvects()  # [(λ, mult, [v1, v2...]), ...]
            # clasificación
            tipo, tau, det, disc = self._clasificar(A_simpl)

            # funciones numéricas para el campo
            fnum, gnum = self._lambdas(fx, gy)

            # limpiar gráfico
            self.ax.clear()

            # ========== Campo vectorial + trayectorias (solo hacia adelante) ==========
            nx, ny = 25, 25
            Xg, Yg = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
            U = fnum(Xg, Yg)
            V = gnum(Xg, Yg)

            self.ax.streamplot(Xg, Yg, U, V, density=1.1, linewidth=1)
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.ax.set_title("Diagrama de Fases (trayectorias hacia adelante)")
            self.ax.grid(True, linestyle='--', alpha=0.5)
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.plot(0, 0, "ko", ms=5)  # equilibrio en el origen

            # Trayectorias hacia adelante
            trajs = self._trayectorias_forward(fnum, gnum, (x_min, x_max), (y_min, y_max), tmax, nseed=ntraj)
            for (Xf, Yf) in trajs:
                self.ax.plot(Xf, Yf, lw=1.2)

            self.canvas.draw()

            # ========== salida texto ==========
            self.output.clear()
            self.output.append("Sistema ingresado:")
            self.output.append(f"  ẋ = {sp.simplify(fx)}")
            self.output.append(f"  ẏ = {sp.simplify(gy)}")
            self.output.append("")
            self.output.append("Matriz del sistema A (Jacobiano constante):")
            self.output.append(str(sp.Matrix(A_simpl)))
            self.output.append("")
            self.output.append(f"Clasificación: {tipo}")
            self.output.append(f"Traza τ = {tau:.6g}")
            self.output.append(f"Determinante det = {det:.6g}")
            self.output.append(f"Discriminante D = τ^2 - 4·det = {disc:.6g}")
            self.output.append("")
            self.output.append("Autovalores y autovectores:")

            for ev in eig_data:
                lam = sp.N(ev[0])
                mult = ev[1]
                vecs = ev[2]
                self.output.append(f"  λ = {lam} (mult. algébrica {mult})")
                if vecs:
                    for j, v in enumerate(vecs, start=1):
                        v = sp.Matrix(v)
                        self.output.append(f"     v{j} = {sp.N(v)}")
                else:
                    self.output.append("     (sin autovectores linealmente independientes; caso jordaniano)")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaseLineal2DWindow()
    window.show()
    sys.exit(app.exec())
