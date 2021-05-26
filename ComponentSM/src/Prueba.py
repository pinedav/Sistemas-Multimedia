from PySide2.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QPushButton, QGraphicsView, QGraphicsItem
from PySide2.QtGui import QBrush, QPen, QFont
from PySide2.QtCore import Qt
import sys


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pyside2 QGraphic View")
        self.setGeometry(300, 200, 640, 520)

        self.create_ui()

        self.show()

    def create_ui(self):

        scene = QGraphicsScene(self)

        greenBrush = QBrush(Qt.red)
        blueBrush = QBrush(Qt.blue)

        blackPen = QPen(Qt.black)
        blackPen.setWidth(1)

        ellipse = scene.addEllipse(0, 0, 20, 20, blackPen, greenBrush)
        ellipse2 = scene.addEllipse(-30, -30, 20, 20, blackPen, blueBrush)

        ellipse.setFlag(QGraphicsItem.ItemIsMovable)
        ellipse2.setFlag(QGraphicsItem.ItemIsMovable)

        self.view = QGraphicsView(scene, self)
        self.view.setGeometry(0, 0, 640, 440)


app = QApplication(sys.argv)
window = Window()
sys.exit(app.exec_())