from PyQt6.QtWidgets import QWidget, QGridLayout, QPushButton, QGraphicsDropShadowEffect
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt

class KeyButton(QPushButton):
    def __init__(self, text, *, radius=10, accent="#7cc5ff"):
        super().__init__(text)
        self.setFixedSize(70, 42)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(f"""
            QPushButton {{
                background: #ffffff;
                color: #0f172a;
                border: 1px solid rgba(0,0,0,0.12);
                border-radius: {radius}px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                border-color: {accent};
                background: #f6f8fb;
            }}
            QPushButton:pressed {{
                background: #eef2f7;
                padding-top: 1px;       
            }}
            QPushButton:disabled {{
                color: rgba(15,23,42,0.35);
                 border-color: rgba(0,0,0,0.08);
                background: #fafafa;
            }}
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)              
        shadow.setOffset(0, 3)                
        shadow.setColor(QColor(0, 0, 0, 140))
        self.setGraphicsEffect(shadow)


class MathKeyboard(QWidget):
    def __init__(self, insert_callback, *, accent="#7cc5ff"):
        super().__init__()
        self.insert_callback = insert_callback
        layout = QGridLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 6, 0, 6)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        botones = [
            ("+", 0, 0), ("-", 0, 1), ("*", 0, 2), ("/", 0, 3),
            ("^", 1, 0), ("sqrt", 1, 1), ("ln", 1, 2), ("exp", 1, 3),
            ("sin", 2, 0), ("cos", 2, 1), ("tan", 2, 2), ("π", 2, 3),
            ("e", 3, 0), ("x", 3, 1), ("(", 3, 2), (")", 3, 3),
        ]

        for texto, fila, col in botones:
            btn = KeyButton(texto, radius=10, accent=accent)
            btn.clicked.connect(lambda checked=False, t=texto: self.insert_callback(t))
            layout.addWidget(btn, fila, col)
