import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QPushButton, QFrame
)
from PyQt6.QtCore import Qt
from Biseccion import BiseccionWindow
from PuntoFijo import PuntoFijoWindow
from Aitken import AitkenWindow
from NewtonRaphson import NewtonRaphsonWindow
from lagrange import LagrangeWindow
from DiferenciasFinitas import DiferenciasFinitasWindow
from NewtonCotes import NewtonCotesWindow
from MontecarloIntegralSimple import MonteCarloWindow
from MontecarloIntegralesDobles import MonteCarloDobleWindow

class MenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador de Integración Numérica")
        self.setGeometry(200, 200, 500, 400)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        
        title = QLabel("Selecciona un método numérico")
        title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            margin: 20px;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #bdc3c7; margin: 10px;")
        layout.addWidget(line)

        metodos = [
            ("Búsqueda binaria de raíces", self.abrir_biseccion),
            ("Método del punto fijo", self.abrir_punto_fijo),
            ("Método de Stephen Aiken", self.abrir_aitken),
            ("Método de Newton Raphson", self.abrir_newton_raphson),
            ("Polinomio interpolante de Lagrange", self.abrir_lagrange),
            ("Derivada por diferencia finita", self.abrir_diferencias_finitas),
            ("Reglas de Newton Cotes", self.abrir_newton_cotes),
            ("Método de Montecarlo para integrales simples", self.abrir_montecarlo_1D),
            ("Método de Montecarlo para integrales dobles", self.abrir_montecarlo_2D),

        ]

        for texto, funcion in metodos:
            boton = QPushButton(texto)
            boton.setFixedHeight(40)
            boton.setStyleSheet("""
                QPushButton {
                    font-size: 14px;
                    background-color: #3498db;
                    color: white;
                    border-radius: 8px;
                    margin: 6px 20px;
                }
                QPushButton:hover { background-color: #2980b9; }
                QPushButton:pressed { background-color: #1f618d; }
            """)
            boton.clicked.connect(funcion)
            layout.addWidget(boton)
        creditos = QLabel("Modelado y Simulación - 2025 - Proyecto desarrollado por María Paula Pierrot, Victoria Abril Novello, Juan Nofal, Marianela Tatiana Tallarico, y Marilyn Nicole Soto.")
        creditos.setStyleSheet("""
            font-size: 12px;
            color: gray;
            margin: 15px;
        """)
        creditos.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(creditos)



    def abrir_biseccion(self):
        """abrir la ventana de biseccion como pop-up"""
        self.biseccion_window = BiseccionWindow()
        self.biseccion_window.show()
        return
        
    def abrir_punto_fijo(self):
        """abrir la ventana de punto fijo como pop-up"""
        self.punto_fijo_window = PuntoFijoWindow()
        self.punto_fijo_window.show()
        return
    
    def abrir_aitken(self):
        """abrir la ventana de Aitken como pop-up"""
        self.aitken_window = AitkenWindow()
        self.aitken_window.show()
        return

    def abrir_newton_raphson(self):
        """abrir la ventana de Newton Raphson como pop-up"""
        self.newton_raphson_window = NewtonRaphsonWindow()
        self.newton_raphson_window.show()
        return
    
    def abrir_lagrange(self):
        """abrir la ventana de Lagrange como pop-up"""
        self.lagrange = LagrangeWindow()
        self.lagrange.show()
        return
    
    def abrir_diferencias_finitas(self):
        """abrir la ventana de diferencias finitas como pop-up"""
        self.diferencias_finitas = DiferenciasFinitasWindow()
        self.diferencias_finitas.show()
        return
    
    def abrir_newton_cotes(self):
        """abrir la ventana de diferencias finitas como pop-up"""
        self.newton_cotes = NewtonCotesWindow()
        self.newton_cotes.show()
        return
    
    def abrir_montecarlo_1D(self):
        """abrir la ventana de diferencias finitas como pop-up"""
        self.montecarlo_1d = MonteCarloWindow()
        self.montecarlo_1d.show()
        return
    
    def abrir_montecarlo_2D(self):
        self.montecarlo_2D = MonteCarloDobleWindow()
        self.montecarlo_2D.show()
        return

# ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    menu = MenuWindow()
    menu.show()
    sys.exit(app.exec())
