from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import sympy as sp

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


# --- Función: Interpolación de Lagrange ---
def lagrange_interpolante_detallado(x_vals, y_vals):
    x = sp.symbols('x')
    n = len(x_vals)
    polinomio = 0
    pasos = []

    for i in range(n):
        Li = 1
        factors = []
        for j in range(n):
            if i != j:
                Li *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
                factors.append(f"(x - {x_vals[j]})/({x_vals[i]} - {x_vals[j]})")
        pasos.append(f"L_{i}(x) = " + " * ".join(factors))
        polinomio += y_vals[i] * Li

    return sp.expand(polinomio), pasos


class LagrangeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interpolación de Lagrange")
        self.setGeometry(100, 100, 1200, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, 1)

        self.table = QTableWidget(3, 2)  # arranca con 3 filas, 2 columnas
        self.table.setHorizontalHeaderLabels(["xᵢ", "yᵢ"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(True)

        default_points = [(0, 1), (1, 3), (2, 2)]
        for row, (x, y) in enumerate(default_points):
            self.table.setItem(row, 0, QTableWidgetItem(str(x)))
            self.table.setItem(row, 1, QTableWidgetItem(str(y)))

        left_panel.addWidget(QLabel("Puntos de interpolación:"))
        left_panel.addWidget(self.table)

        self.add_row_btn = QPushButton("Agregar punto")
        style_primary_button(self.add_row_btn) 
        self.add_row_btn.clicked.connect(self.agregar_fila)
        left_panel.addWidget(self.add_row_btn)

        self.calc_btn = QPushButton("Calcular polinomio")
        style_primary_button(self.calc_btn) 
        self.calc_btn.clicked.connect(self.calcular)
        left_panel.addWidget(self.calc_btn)

        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, 3)

        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        right_panel.addWidget(self.canvas)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        right_panel.addWidget(self.output)

    def agregar_fila(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(""))
        self.table.setItem(row, 1, QTableWidgetItem(""))

    def leer_puntos(self):
        x_vals, y_vals = [], []
        for row in range(self.table.rowCount()):
            x_item = self.table.item(row, 0)
            y_item = self.table.item(row, 1)

            if x_item and y_item and x_item.text() and y_item.text():
                try:
                    x = float(x_item.text())
                    y = float(y_item.text())
                    x_vals.append(x)
                    y_vals.append(y)
                    x_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    y_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                except ValueError:
                    raise ValueError(f"Entrada inválida en fila {row+1}")
        return x_vals, y_vals

    def calcular(self):
        try:
            x_vals, y_vals = self.leer_puntos()
            if len(x_vals) < 2:
                raise ValueError("Se necesitan al menos 2 puntos para interpolar.")

            polinomio, pasos = lagrange_interpolante_detallado(x_vals, y_vals)
            f_lagrange = sp.lambdify(sp.symbols('x'), polinomio, "numpy")

            # --- gráfico ---
            self.ax.clear()
            X = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 400)
            Y = f_lagrange(X)

            self.ax.plot(X, Y, "b-", label="Polinomio interpolante")
            self.ax.plot(x_vals, y_vals, "ro", label="Puntos dados")
            self.ax.set_title("Interpolación de Lagrange")
            self.ax.legend()
            self.ax.grid(True)

            # Mostrar polinomio en LaTeX sobre el gráfico
            self.ax.text(
                0.05, 0.95,
                f"$P(x) = {sp.latex(polinomio)}$",
                transform=self.ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
            )

            self.canvas.draw()

            # ---
            self.output.clear()
            self.output.append("Construcción del polinomio de Lagrange:\n")
            for paso in pasos:
                self.output.append(paso)
            self.output.append("\nPolinomio final expandido:")
            self.output.append(str(polinomio))

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# --- 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LagrangeWindow()
    window.show()
    sys.exit(app.exec())
