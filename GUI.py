import os
import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox

import design
from captions_generator import get_predict
from feture_extraction import extracting_features_from_image
from text_data_preparation import prepare_text_data
from training import start_fit_model, start_lite_train


class MainApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pathImageButton_4.clicked.connect(self.chose_image_path)
        self.pathModelButton_5.clicked.connect(self.chose_model_path)
        self.tariningPushButton.clicked.connect(self.training_start)
        self.liteTrainingPushButton_2.clicked.connect(self.lite_training_start)
        self.predictionButton_3.clicked.connect(self.prediction)

    def chose_image_path(self):
        """
        функция использутеся для описания функционала кнопки "Указать путь к изображению"
        и вывода выбранного изображения в окно предпросмотра
        """
        self.imagePathLineEdit.clear()
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите изображение",
                                                     filter="*.jpg")
        if os.path.exists(path[0]):
            self.imagePathLineEdit.setText(path[0])

        scene = QtWidgets.QGraphicsScene()
        pixmap = QPixmap(path[0])
        pixmap = pixmap.scaled(478, 348)
        scene.addPixmap(pixmap)
        self.imagePreView.setScene(scene)
        self.imagePreView.show()

    def chose_model_path(self):
        """
        функция использутеся для описания функционала кнопки "Указать путь к модели"
        """
        self.modelPathLineEdit_2.clear()
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите модель",
                                                     filter="*.h5")
        if os.path.exists(path[0]):
            self.modelPathLineEdit_2.setText(path[0])

    def training_start(self):
        """
        функция использутеся для описания функционала кнопки "Тренировка",
        запускает обучение модели в соответствии с указанными параметрами,
        ведет лог происходящих событий
        """
        QMessageBox.about(self, "ПРЕДУПРЕЖДЕНИЕ", "Во время тренировки рабочая область программы может быть недоступна.")
        if self.checkBox.checkState():
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText("Подготовка текстовых данных начата! Это может занять несколько минут...\n")
            prepare_text_data()
            self.LogBrowser.insertPlainText("Подготовка текстовых данных завершена!\n\n")
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText("Извлечение признаков из тренировочного набора данных начато! Это может занять несколько минут...\n")
            extracting_features_from_image("Flickr8k_Dataset/Flicker8k_Dataset")
            self.LogBrowser.insertPlainText("Извлечение признаков завершено!\n\n")
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText("Обучение начато. Пожалуйста подождите!\n")
            start_fit_model(20)
            self.LogBrowser.insertPlainText("Обучение завершено. Полный лог событий можно увидеть в фале model.log.\n\n")
            QCoreApplication.processEvents()
        else:
            if not os.path.exists("captions.txt"):
                QCoreApplication.processEvents()
                self.LogBrowser.insertPlainText(
                    "Подготовка текстовых данных начата! Это может занять несколько минут...\n")
                prepare_text_data()
                self.LogBrowser.insertPlainText(
                    "Подготовка текстовых данных завершена!\n\n")
                QCoreApplication.processEvents()
            else:
                QCoreApplication.processEvents()
                self.LogBrowser.insertPlainText("Пропуск этапа подготовки текстовых данных. "
                                                "(файл captions.txt уже существует)\n\n")
                QCoreApplication.processEvents()
            if not os.path.exists("features.pkl"):
                QCoreApplication.processEvents()
                self.LogBrowser.insertPlainText(
                    "Извлечение признаков из тренировочного набора данных начато! Это может занять несколько минут...\n")
                extracting_features_from_image(
                    "Flickr8k_Dataset/Flicker8k_Dataset")
                self.LogBrowser.insertPlainText(
                    "Извлечение признаков завершено!\n\n")
                QCoreApplication.processEvents()
            else:
                QCoreApplication.processEvents()
                self.LogBrowser.insertPlainText("Пропуск этапа извлечения признаков. "
                                                "(файл features.pkl уже существует)\n\n")
                QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText(
                "Обучение начато. Пожалуйста подождите!\n")
            start_fit_model(20)
            self.LogBrowser.insertPlainText(
                "Обучение завершено. Полный лог событий можно увидеть в файле model.log.\n\n")
            QCoreApplication.processEvents()

    def lite_training_start(self):
        """
        функция использутеся для описания функционала кнопки "Облегчённый режим тренировки",
        запускает облегченное обучение модели в соответствии с указанными параметрами,
        ведет лог происходящих событий
        """
        QMessageBox.about(self, "ПРЕДУПРЕЖДЕНИЕ", "Во время тренировки рабочая область программы может быть недоступна.")
        if self.checkBox.checkState():
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText(
                "Подготовка текстовых данных начата! Это может занять несколько минут...\n")
            prepare_text_data()
            self.LogBrowser.insertPlainText(
                "Подготовка текстовых данных завершена!\n\n")
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText(
                "Извлечение признаков из тренировочного набора данных начато! Это может занять несколько минут...\n")
            extracting_features_from_image(
                "Flickr8k_Dataset/Flicker8k_Dataset")
            self.LogBrowser.insertPlainText(
                "Извлечение признаков завершено!\n\n")
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText(
                "Обучение начато. Пожалуйста подождите!\n")
            start_lite_train(20)
            self.LogBrowser.insertPlainText(
                "Обучение завершено. Полный лог событий можно увидеть в файле model.log.\n\n")
            QCoreApplication.processEvents()
        else:
            if not os.path.exists("captions.txt"):
                QCoreApplication.processEvents()
                self.LogBrowser.insertPlainText(
                    "Подготовка текстовых данных начата! Это может занять несколько минут...\n")
                prepare_text_data()
                self.LogBrowser.insertPlainText(
                    "Подготовка текстовых данных завершена!\n\n")
                QCoreApplication.processEvents()
            else:
                self.LogBrowser.insertPlainText("Пропуск этапа подготовки текстовых данных. "
                                                "(файл captions.txt уже существует)\n\n")
                QCoreApplication.processEvents()
            if not os.path.exists("features.pkl"):
                self.LogBrowser.insertPlainText(
                    "Извлечение признаков из тренировочного набора данных начато! Это может занять несколько минут...\n")
                QApplication.processEvents()
                extracting_features_from_image(
                    "Flickr8k_Dataset/Flicker8k_Dataset")
                self.LogBrowser.insertPlainText(
                    "Извлечение признаков завершено!\n\n")
                QCoreApplication.processEvents()
            else:
                self.LogBrowser.insertPlainText("Пропуск этапа извлечения признаков. "
                                                "(файл features.pkl уже существует)\n\n")
                QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText(
                "Обучение начато. Пожалуйста подождите!\n")
            start_lite_train(20)
            self.LogBrowser.insertPlainText(
                "Обучение завершено. Полный лог событий можно увидеть в файле model.log.\n\n")
            QCoreApplication.processEvents()

    def prediction(self):
        """
        функция использутеся для описания функционала кнопки "Выполнить описание",
        производит вывод в приложение сгенерированного текстового описания изображения
        """
        filepath = self.imagePathLineEdit.text()
        modelpath = self.modelPathLineEdit_2.text()
        caption = get_predict(filepath, modelpath)
        self.LogBrowser.insertPlainText("Описание для изображения {} при указанной модели {}: \n{}\n\n".format(filepath.split("/")[-1], modelpath.split("/")[-1], caption))


def main():
    self = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    self.exec_()


if __name__ == '__main__':
    main()
