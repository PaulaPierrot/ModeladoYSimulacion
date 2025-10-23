from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QGraphicsDropShadowEffect
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
        self.setWindowTitle("Sistemas Dinámicos - Campo de Pendientes y Estabilidad")
        self.setGeometry(100, 100, 1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- inputs ---
        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)

        # Ejemplo con múltiples puntos de equilibrio: y*(1-y)*(y-2)
        # O: (y-1)*(y-5) para probar la imagen de referencia (parábola)
        self.eq_input = QLineEdit("(y-1)*(y-5)") 
        self.var_input = QLineEdit("y")
        self.range_input = QLineEdit("0,6")  # rango en eje Y

        form_layout.addWidget(QLabel("dy/dt = "))
        form_layout.addWidget(self.eq_input)
        form_layout.addWidget(QLabel("Variable:"))
        form_layout.addWidget(self.var_input)
        form_layout.addWidget(QLabel("Rango y:"))
        form_layout.addWidget(self.range_input)

        # --- botón ---
        self.calc_btn = QPushButton("Analizar sistema")
        style_primary_button(self.calc_btn)
        self.calc_btn.clicked.connect(self.analizar)
        layout.addWidget(self.calc_btn)

        # --- gráficos (2 paneles) ---
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # --- salida de texto ---
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    def analizar(self):
        try:
            expr_str = self.eq_input.text()
            var_str = self.var_input.text().strip()
            rango_str = self.range_input.text()

            y_min_lim, y_max_lim = [float(x) for x in rango_str.split(",")]

            # Símbolos para SymPy
            t = sp.Symbol("t")
            y_sym_func = sp.Function(var_str)(t) 
            y_sym = sp.Symbol(var_str)
            
            # Expresiones
            expr_func = sp.sympify(expr_str, locals={"pi": sp.pi, "E": sp.E, var_str: y_sym_func})
            expr_y = sp.sympify(expr_str, locals={var_str: y_sym})
            f = sp.lambdify(y_sym, expr_y, "numpy")

            # --- Resolución Analítica (Corregido para manejar listas de soluciones) ---
            try:
                sol_raw = sp.dsolve(sp.Eq(sp.diff(y_sym_func, t), expr_func), y_sym_func)
                
                # Manejar si dsolve retorna una lista (caso común para múltiples soluciones o formas)
                if isinstance(sol_raw, list):
                    sol = sol_raw[0] # Tomar la primera solución
                else:
                    sol = sol_raw
                    
            except Exception:
                sol = "No se pudo encontrar una solución analítica explícita"

            # --- Puntos de Equilibrio y Derivada ---
            eq_points_sym = sp.solve(sp.Eq(expr_y, 0), y_sym)
            eq_points = sorted([float(p) for p in eq_points_sym if p.is_real])
            dfdy = sp.diff(expr_y, y_sym)

            # limpiar gráficos
            self.ax1.clear()
            self.ax2.clear()

            # === 1) Campo de pendientes en (t,y) ===
            y_vals = np.linspace(y_min_lim, y_max_lim, 25)
            t_vals = np.linspace(-5, 10, 25) # Rango t para trayectorias
            T, Y = np.meshgrid(t_vals, y_vals)
            DY = f(Y)
            DT = np.ones_like(DY)
            N = np.sqrt(DT**2 + DY**2)
            self.ax1.quiver(T, Y, DT/N, DY/N, angles="xy", scale_units='xy', scale=1, color='gray')

            # líneas de fase (soluciones aproximadas)
            for y0 in np.linspace(y_min_lim, y_max_lim, 6):
                dt = 0.05
                # Integración hacia adelante (t > 0)
                ys_fwd = [y0]
                for _ in range(200):
                    next_y = ys_fwd[-1] + f(ys_fwd[-1]) * dt
                    if next_y > y_max_lim * 1.05 or next_y < y_min_lim * 1.05: break
                    ys_fwd.append(next_y)
                
                # Integración hacia atrás (t < 0)
                ys_bwd = [y0]
                for _ in range(200):
                    prev_y = ys_bwd[-1] + f(ys_bwd[-1]) * (-dt) 
                    if prev_y > y_max_lim * 1.05 or prev_y < y_min_lim * 1.05: break
                    ys_bwd.append(prev_y)

                t_fwd = np.linspace(0, len(ys_fwd)*dt, len(ys_fwd))
                t_bwd = np.linspace(-len(ys_bwd)*dt, 0, len(ys_bwd))
                
                self.ax1.plot(t_fwd, ys_fwd, "r-", alpha=0.8, lw=1.5)
                self.ax1.plot(t_bwd, list(reversed(ys_bwd)), "r-", alpha=0.8, lw=1.5)
                
            self.ax1.set_xlim(np.min(t_vals), np.max(t_vals))
            self.ax1.set_ylim(y_min_lim, y_max_lim)
            self.ax1.set_title("Campo de Pendientes y Trayectorias")
            self.ax1.set_xlabel("t")
            self.ax1.set_ylabel(var_str)
            self.ax1.grid(True, linestyle='--', alpha=0.6)

            # mostrar la solución analítica en el gráfico
            if isinstance(sol, str):
                sol_text = sol
            else:
                sol_text = f"y(t) = {sp.latex(sol.rhs)}"
                
            self.ax1.text(
                0.05, 0.95, f"$ {sol_text} $",
                transform=self.ax1.transAxes,
                fontsize=12, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            # === 2) Diagrama de fases 1D (con flujo en intervalos y flechas de estabilidad) ===
            y_range = np.linspace(y_min_lim, y_max_lim, 400)
            fy = f(y_range)
            self.ax2.axhline(0, color="k", lw=1)
            self.ax2.plot(y_range, fy, "b", label=f"f({var_str})")
            
            interval_points = [y_min_lim] + eq_points + [y_max_lim]

            resultados = []
            
            # Análisis de estabilidad y graficado de puntos de equilibrio
            # y las flechas locales de estabilidad
            delta = 0.03 * (y_max_lim - y_min_lim) # Tamaño de las flechitas locales
            
            for eq_val in eq_points:
                dfdy_val = dfdy.subs(y_sym, eq_val)
                estabilidad = sp.sign(dfdy_val)
                st = "Neutro"
                color = "k"
                marker = "o" # Default circular
                
                # Tolerancia para el análisis de estabilidad (df/dy = 0)
                if not dfdy_val.is_number or abs(dfdy_val) < 1e-9:
                    st = "Neutro / Semiestable"
                    color = "k"
                    marker = "s" # Cuadrado para semiestable/neutro
                elif estabilidad < 0:
                    st = "Estable (Atractor)"
                    color = "g"
                    marker = "o"
                elif estabilidad > 0:
                    st = "Inestable (Repulsor)"
                    color = "r"
                    marker = "o"

                # Marcar punto de equilibrio
                self.ax2.plot(eq_val, 0, marker, color=color, markersize=10, zorder=5)
                self.ax2.text(eq_val, 0.05 * (y_max_lim - y_min_lim), f"{eq_val:.2f}", ha="center", color=color, fontsize=10)
                resultados.append(f"Equilibrio en {var_str}={eq_val:.4f}: {st}")

                # --- Flechas locales de estabilidad ---
                arrow_properties = dict(arrowstyle="->", color=color, lw=2, mutation_scale=15)
                
                if estabilidad < 0:  # Estable -> flechas hacia el punto
                    self.ax2.annotate("", xy=(eq_val, 0), xytext=(eq_val - delta, 0),
                                      arrowprops=arrow_properties)
                    self.ax2.annotate("", xy=(eq_val, 0), xytext=(eq_val + delta, 0),
                                      arrowprops=arrow_properties)
                elif estabilidad > 0:  # Inestable -> flechas salen del punto
                    self.ax2.annotate("", xy=(eq_val - delta, 0), xytext=(eq_val, 0),
                                      arrowprops=arrow_properties)
                    self.ax2.annotate("", xy=(eq_val + delta, 0), xytext=(eq_val, 0),
                                      arrowprops=arrow_properties)
                # Para puntos neutros, no se dibujan flechas locales de estabilidad


            # --- Graficado del Flujo en los Intervalos (flechas más grandes) ---
            for i in range(len(interval_points) - 1):
                y1 = interval_points[i]
                y2 = interval_points[i+1]
                
                # Tomar un punto de prueba en el medio
                y_test = (y1 + y2) / 2
                
                # Asegurar que el punto de prueba esté dentro del rango del gráfico
                if y_test < y_min_lim or y_test > y_max_lim: continue
                
                # Si el intervalo es demasiado pequeño o centrado en un punto de equilibrio, saltar
                if abs(y2 - y1) < 1e-3 or (abs(f(y_test)) < 1e-9 and y_test not in eq_points): continue

                # Evaluar f(y) para determinar la dirección
                fy_test = f(y_test)
                
                if abs(fy_test) < 1e-9: continue # Evitar intervalos planos con flujo nulo

                # Posición y tamaño de la flecha
                arrow_x = y_test
                length = (y2 - y1) * 0.7 
                
                if fy_test > 0: # Creciente (hacia la derecha)
                    arrow_color = "darkblue"
                    # Dibujar flecha (x_start, y_start, dx, dy)
                    self.ax2.arrow(arrow_x - length/2, 0, length, 0, 
                                   head_width=0.02 * (y_max_lim - y_min_lim), # Ajustado tamaño cabeza
                                   head_length=0.03 * (y_max_lim - y_min_lim), # Ajustado tamaño cabeza
                                   fc=arrow_color, ec=arrow_color, linewidth=1, zorder=3) # Ancho de línea ajustado
                else: # Decreciente (hacia la izquierda)
                    arrow_color = "purple"
                    # Dibujar flecha (x_start, y_start, dx, dy)
                    self.ax2.arrow(arrow_x + length/2, 0, -length, 0, 
                                   head_width=0.02 * (y_max_lim - y_min_lim), # Ajustado tamaño cabeza
                                   head_length=0.03 * (y_max_lim - y_min_lim), # Ajustado tamaño cabeza
                                   fc=arrow_color, ec=arrow_color, linewidth=1, zorder=3) # Ancho de línea ajustado


            # Configuración final del gráfico
            self.ax2.set_title("Diagrama de Fases 1D")
            self.ax2.set_xlabel(var_str)
            self.ax2.set_ylabel(f"f({var_str}) = dy/dt")
            self.ax2.set_xlim(y_min_lim, y_max_lim)
            self.ax2.grid(True, linestyle='--', alpha=0.6)
            self.ax2.legend()

            self.canvas.draw()

            # salida de texto
            self.output.clear()
            self.output.append("Resultados del análisis:\n")
            for r in resultados:
                self.output.append(r)

            self.output.append("\nResolución analítica de la EDO:\n")
            if isinstance(sol, str):
                 self.output.append(sol)
            else:
                self.output.append(f"y(t) = ${sp.latex(sol.rhs)}$")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SistemaDinamicoWindow()
    window.show()
    sys.exit(app.exec())
