import os
import pandas as pd
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from config import BASE_DIR
import sys


class Labeler:
    def __init__(self):

        try:
            self.df = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/dataset.csv'))
            self.image_num = self.df['image'].max() + 1
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=['image', 'label'])
            self.image_num = 1

        self.images_amount = len(os.listdir(os.path.join(BASE_DIR, 'data/raw/images')))

        app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setGeometry(400, 100, 800, 800)
        self.window.setWindowTitle('Image labeler')

        self.button_urbanized = QPushButton(self.window)
        self.button_urbanized.setText('Urbanized')
        self.button_urbanized.setGeometry(200, 600, 150, 150)
        self.button_urbanized.clicked.connect(lambda: self.classify(1))

        self.button_not_urbanized = QPushButton(self.window)
        self.button_not_urbanized.setText('Not urbanized')
        self.button_not_urbanized.setGeometry(400, 600, 150, 150)
        self.button_not_urbanized.clicked.connect(lambda: self.classify(0))

        self.button_back = QPushButton(self.window)
        self.button_back.setText('Back')
        self.button_back.setGeometry(50, 600, 100, 50)
        self.button_back.clicked.connect(self.back)

        self.button_save = QPushButton(self.window)
        self.button_save.setText('Save')
        self.button_save.setGeometry(600, 600, 100, 50)
        self.button_save.clicked.connect(self.save)

        self.image = QLabel(self.window)
        self.image.setGeometry(230, 200, 300, 300)

        self.img = Image.open(os.path.join(BASE_DIR, f'data/raw/images/{self.image_num}.tif'))
        imag = self.img.resize((300, 300))
        qimage = QImage(imag.tobytes(), imag.width, imag.height, QImage.Format_RGB888)
        self.image.setPixmap(QPixmap.fromImage(qimage))

        self.window.show()
        sys.exit(app.exec_())

    def classify(self, label):
        new_row = [self.image_num, label]
        self.df.loc[len(self.df)] = new_row

        self.image_num += 1
        try:
            self.img = Image.open(os.path.join(BASE_DIR, f'data/raw/images/{self.image_num}.tif'))
        except FileNotFoundError:
            self.save()
            print('All images labeled, data seved to dataset.csv...')
            sys.exit()

        imag = self.img.resize((300, 300))
        qimage = QImage(imag.tobytes(), imag.width, imag.height, QImage.Format_RGB888)
        self.image.setPixmap(QPixmap.fromImage(qimage))

        print(f'{self.image_num}/{self.images_amount}')
        self.save()

    def back(self):
        self.image_num -= 1

        self.img = Image.open(os.path.join(BASE_DIR, f'data/raw/images/{self.image_num}.tif'))
        imag = self.img.resize((300, 300))
        qimage = QImage(imag.tobytes(), imag.width, imag.height, QImage.Format_RGB888)
        self.image.setPixmap(QPixmap.fromImage(qimage))

        self.df.drop(self.df.tail(1).index, inplace=True)

    def save(self):
        self.df.to_csv(os.path.join(BASE_DIR, 'data/raw/dataset.csv'), index=False)

if __name__ == "__main__":
    Labeler()
