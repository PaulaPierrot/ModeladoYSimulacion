from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QGraphicsDropShadowEffect
)
import sys
import numpy as np
import matplotlib.pyplot as plt
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


class LogisticoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modelo Logístico Discreto - Crecimiento Poblacional")
        self.setGeometry(100, 100, 1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- inputs ---
        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)

        self.n0_input = QLineEdit("10")
        self.lam_input = QLineEdit("2.5")
        self.k_input = QLineEdit("100")
        self.steps_input = QLineEdit("50")

        form_layout.addWidget(QLabel("N₀:"))
        form_layout.addWidget(self.n0_input)
        form_layout.addWidget(QLabel("λ:"))
        form_layout.addWidget(self.lam_input)
        form_layout.addWidget(QLabel("K:"))
        form_layout.addWidget(self.k_input)
        form_layout.addWidget(QLabel("Iteraciones:"))
        form_layout.addWidget(self.steps_input)

        # --- botón ---
        self.calc_btn = QPushButton("Simular")
        style_primary_button(self.calc_btn)
        self.calc_btn.clicked.connect(self.simular)
        layout.addWidget(self.calc_btn)

        # --- gráficos (2 paneles) ---
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # --- salida de texto ---
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    def simular(self):
        try:
            N0 = float(self.n0_input.text())
            lam = float(self.lam_input.text())
            K = float(self.k_input.text())
            steps = int(self.steps_input.text())

            # modelo logístico discreto
            N = [N0]
            for _ in range(steps):
                Nn = lam * N[-1] * (1 - N[-1] / K)
                N.append(Nn)

            # limpiar gráficos
            self.ax1.clear()
            self.ax2.clear()

            # === 1) Evolución temporal ===
            self.ax1.plot(range(len(N)), N, "bo-", label="N(t)")
            self.ax1.axhline(K, color="gray", ls="--", label=f"K={K}")
            self.ax1.set_title("Evolución de la población")
            self.ax1.set_xlabel("Iteración (n)")
            self.ax1.set_ylabel("Población N")
            self.ax1.legend()
            self.ax1.grid(True)

            # === 2) Diagrama de fases discreto ===
            Nn = np.array(N[:-1])
            Nn1 = np.array(N[1:])
            self.ax2.plot(Nn, Nn1, "ro-", alpha=0.7, label="Iteraciones")
            yline = np.linspace(0, K, 200)
            self.ax2.plot(yline, lam * yline * (1 - yline / K), "b", label="f(N)")
            self.ax2.plot(yline, yline, "k--", label="Nₙ₊₁ = Nₙ")
            self.ax2.set_title("Diagrama de fases discreto")
            self.ax2.set_xlabel("Nₙ")
            self.ax2.set_ylabel("Nₙ₊₁")
            self.ax2.legend()
            self.ax2.grid(True)

            self.canvas.draw()

            # === salida de texto: equilibrios ===
            self.output.clear()
            self.output.append("Análisis del modelo logístico:\n")

            # puntos de equilibrio: N=0 y N=K(1-1/λ)
            eq_points = [0, K * (1 - 1/lam)] if lam > 1 else [0]

            for eq in eq_points:
                if eq < 0: 
                    continue
                # derivada f'(N) = λ(1 - 2N/K)
                stability = lam * (1 - 2*eq/K)
                if abs(stability) < 1:
                    st = "Estable"
                else:
                    st = "Inestable"
                self.output.append(f"Equilibrio en N={eq:.4f} → {st}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LogisticoWindow()
    window.show()
    sys.exit(app.exec())
