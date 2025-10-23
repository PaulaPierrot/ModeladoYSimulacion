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


# --- estilos para botones ---
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


class SistemaDinamicoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistemas Dinámicos y Bifurcaciones")
        self.setGeometry(100, 100, 1600, 1000)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- inputs ---
        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)

        self.bif_combo = QComboBox()
        self.bif_combo.addItems(["Transcrítica", "Silla-Nodo", "Tridente", "Personalizado"])
        self.bif_combo.currentIndexChanged.connect(self.actualizar_input)

        self.eq_input = QLineEdit("r*y - y**2")
        self.var_input = QLineEdit("y")
        self.range_input = QLineEdit("-3,3")  # rango en eje Y
        self.r_input = QLineEdit("1")  # valor inicial de r

        # nuevos inputs para r_neg y r_pos
        self.r_neg_input = QLineEdit("-1")
        self.r_pos_input = QLineEdit("1")

        form_layout.addWidget(QLabel("Tipo:"))
        form_layout.addWidget(self.bif_combo)
        form_layout.addWidget(QLabel("dy/dt ="))
        form_layout.addWidget(self.eq_input)
        form_layout.addWidget(QLabel("Variable:"))
        form_layout.addWidget(self.var_input)
        form_layout.addWidget(QLabel("Rango y:"))
        form_layout.addWidget(self.range_input)
        form_layout.addWidget(QLabel("r inicial:"))
        form_layout.addWidget(self.r_input)
        form_layout.addWidget(QLabel("r<0:"))
        form_layout.addWidget(self.r_neg_input)
        form_layout.addWidget(QLabel("r>0:"))
        form_layout.addWidget(self.r_pos_input)

        # --- botón ---
        self.calc_btn = QPushButton("Analizar sistema")
        style_primary_button(self.calc_btn)
        self.calc_btn.clicked.connect(self.analizar)
        layout.addWidget(self.calc_btn)

        # --- gráficos (5 paneles) ---
        self.figure = plt.figure(figsize=(16, 10))
        gs = self.figure.add_gridspec(2, 3, hspace=0.6)

        self.ax1 = self.figure.add_subplot(gs[0, 0])  # campo de pendientes
        self.ax_bif = self.figure.add_subplot(gs[0, 1:])  # bifurcación
        self.ax2 = self.figure.add_subplot(gs[1, 0])  # fase r<0
        self.ax3 = self.figure.add_subplot(gs[1, 1])  # fase r=0
        self.ax4 = self.figure.add_subplot(gs[1, 2])  # fase r>0

        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # --- salida de texto ---
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    def actualizar_input(self):
        tipo = self.bif_combo.currentText()
        if tipo == "Transcrítica":
            self.eq_input.setText("r*y - y**2")
        elif tipo == "Silla-Nodo":
            self.eq_input.setText("r + y**2")
        elif tipo == "Tridente":
            self.eq_input.setText("r*y - y**3")
        else:
            self.eq_input.setText("(y-1)*(y-5)")

    def analizar(self):
        try:
            var_str = self.var_input.text().strip()
            y_min_lim, y_max_lim = [float(x) for x in self.range_input.text().split(",")]
            r_val = float(self.r_input.text())

            # Leer r_neg y r_pos
            r_neg = float(self.r_neg_input.text())
            r_pos = float(self.r_pos_input.text())
            if not (r_neg < 0 and r_pos > 0):
                QMessageBox.critical(self, "Error", "Debe ingresar r_neg < 0 y r_pos > 0")
                return

            # Definir símbolos
            y = sp.Symbol(var_str)
            r = sp.Symbol("r")

            # Expresión ingresada
            expr_str = self.eq_input.text()
            expr = sp.sympify(expr_str, locals={"y": y, "r": r, "pi": sp.pi, "E": sp.E})

            # Función evaluable
            f = sp.lambdify((y, r), expr, "numpy")

            # limpiar gráficos
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            self.ax_bif.clear()

            resultados = []

            # ========== 1) Campo de pendientes ==========
            y_vals = np.linspace(y_min_lim, y_max_lim, 25)
            t_vals = np.linspace(-5, 10, 25)
            T, Y = np.meshgrid(t_vals, y_vals)
            DY = f(Y, r_val)
            DT = np.ones_like(DY)
            N = np.sqrt(DT**2 + DY**2)
            self.ax1.quiver(T, Y, DT/N, DY/N, angles="xy", scale_units='xy', scale=1, color='gray')
            self.ax1.set_xlim(np.min(t_vals), np.max(t_vals))
            self.ax1.set_ylim(y_min_lim, y_max_lim)
            self.ax1.set_title(f"Campo de Pendientes (r={r_val})")
            self.ax1.set_xlabel("t")
            self.ax1.set_ylabel(var_str)
            self.ax1.grid(True, linestyle='--', alpha=0.6)

            # Helper para diagramas de fases
            def graficar_fase(ax, case_r, color, titulo):
                expr_r = expr.subs(r, case_r)
                f_case = sp.lambdify(y, expr_r, "numpy")
                yy = np.linspace(y_min_lim, y_max_lim, 400)
                fy = f_case(yy)
                ax.plot(yy, fy, color, label=f"r={case_r:.2f}")

                dfdy = sp.diff(expr_r, y)
                eq_points = sp.solve(sp.Eq(expr_r, 0), y)
                for eq in eq_points:
                    if eq.is_real:
                        stab = dfdy.subs(y, eq)
                        if stab < 0:
                            ax.plot(float(eq), 0, "go", markersize=8)
                            resultados.append(f"Equilibrio en y={eq:.3f}, r={case_r:.2f}: Estable")
                        elif stab > 0:
                            ax.plot(float(eq), 0, "ro", markersize=8)
                            resultados.append(f"Equilibrio en y={eq:.3f}, r={case_r:.2f}: Inestable")
                        else:
                            ax.plot(float(eq), 0, "ks", markersize=8)
                            resultados.append(f"Equilibrio en y={eq:.3f}, r={case_r:.2f}: Neutro")

                ax.axhline(0, color="k", lw=1)
                ax.set_title(titulo)
                ax.set_xlabel(var_str)
                ax.set_ylabel(f"f({var_str})")
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()

            # ========== 2) Fases separadas ==========
            graficar_fase(self.ax2, r_neg, "r", f"Diagrama de fases (r={r_neg})")
            graficar_fase(self.ax3, 0, "g", "Diagrama de fases (r=0)")
            graficar_fase(self.ax4, r_pos, "b", f"Diagrama de fases (r={r_pos})")

            # ========== 3) Diagrama de bifurcación ==========
            r_vals = np.linspace(-5, 5, 300)
            eq_points = sp.solve(sp.Eq(expr, 0), y)
            dfdy = sp.diff(expr, y)

            for eq in eq_points:
                y_func = sp.lambdify(r, eq, "numpy")
                try:
                    y_vals = np.array(y_func(r_vals), dtype=float)
                    if y_vals.ndim == 0:
                        y_vals = np.full_like(r_vals, float(y_vals))
                except Exception:
                    continue

                dfdy_func = sp.lambdify((y, r), dfdy, "numpy")
                stab_vals = dfdy_func(y_vals, r_vals)

                stable_mask = stab_vals < 0
                unstable_mask = stab_vals > 0

                if np.any(stable_mask):
                    self.ax_bif.plot(r_vals[stable_mask], y_vals[stable_mask], "b-")
                if np.any(unstable_mask):
                    self.ax_bif.plot(r_vals[unstable_mask], y_vals[unstable_mask], "r--")

            self.ax_bif.axhline(0, color="k", lw=1)
            self.ax_bif.axvline(0, color="orange", linestyle=":", lw=2, label="Punto crítico r=0")
            self.ax_bif.set_title("Diagrama de Bifurcación")
            self.ax_bif.set_xlabel("r")
            self.ax_bif.set_ylabel("y*")
            self.ax_bif.grid(True, linestyle='--', alpha=0.6)
            self.ax_bif.legend()

            self.canvas.draw()

            # salida texto
            self.output.clear()
            self.output.append("Ecuación usada: dy/dt = " + str(expr))
            self.output.append("")
            for rtxt in resultados:
                self.output.append(rtxt)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SistemaDinamicoWindow()
    window.show()
    sys.exit(app.exec())
