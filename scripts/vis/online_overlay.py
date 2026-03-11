import sys, os
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QWidget

class TransparentImageViewer(QWidget):
    def __init__(self, image_folder):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.7)

        self.image_folder = image_folder
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg", ".png"))])
        self.index = 0

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        self.load_image()

        self.move(300, 300)
        self.resize(self.label.pixmap().size())
        self.dragging = False

    def load_image(self):
        if not self.image_files:
            return
        filename = self.image_files[self.index]
        img_path = os.path.join(self.image_folder, filename)
        pixmap = QPixmap(img_path)

        # Add orange transparent overlay
        painter = QPainter(pixmap)
        painter.fillRect(pixmap.rect(), QColor(255, 165, 0, 100))  # semi-transparent orange
        
        # Draw filename in white text
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(self.font())  # or use QFont("Arial", 16)
        painter.drawText(10, 30, filename)  # (x, y, text)

        painter.end()

        self.label.setPixmap(pixmap)
        self.resize(pixmap.size())

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Down, Qt.Key_S):
            self.index = (self.index + 1) % len(self.image_files)
            self.load_image()
        elif event.key() in (Qt.Key_Up, Qt.Key_W):
            self.index = (self.index - 1) % len(self.image_files)
            self.load_image()
        elif event.key() == Qt.Key_Escape:
            QApplication.quit()  # this fully quits the app

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(event.globalPos() - self.drag_pos)

    def mouseReleaseEvent(self, event):
        self.dragging = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    if len(sys.argv) != 2:
        print("Usage: python transparent_viewer.py /path/to/images")
        sys.exit(1)
    viewer = TransparentImageViewer(sys.argv[1])
    viewer.show()
    sys.exit(app.exec_())
