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
        self.imagePathLineEdit.clear()
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Choose image",
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
        self.modelPathLineEdit_2.clear()
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Choose model file",
                                                     filter="*.h5")
        if os.path.exists(path[0]):
            self.modelPathLineEdit_2.setText(path[0])

    def training_start(self):
        QMessageBox.about(self, "INFO", "While training takes place, the work area may not be available.")
        if self.checkBox.checkState():
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText("Preparing text data was started! It may takes several minutes...\n")
            prepare_text_data()
            self.LogBrowser.insertPlainText("Preparing text data was finished!\n\n")
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText("Extracting features from training dataset was started. It may takes several minutes...\n")
            extracting_features_from_image("Flickr8k_Dataset/Flicker8k_Dataset")
            self.LogBrowser.insertPlainText("Extracting features was finished!\n\n")
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText("Model fit was started. Please wait!\n")
            start_fit_model(10)
            self.LogBrowser.insertPlainText("Fit was finished. Full log you can see in model.log file.\n\n")
            QCoreApplication.processEvents()
        else:
            if not os.path.exists("captions.txt"):
                QCoreApplication.processEvents()
                self.LogBrowser.insertPlainText(
                    "Preparing text data was started! It may takes several minutes...\n")
                prepare_text_data()
                self.LogBrowser.insertPlainText(
                    "Preparing text data was finished!\n\n")
                QCoreApplication.processEvents()
            else:
                QCoreApplication.processEvents()
                self.LogBrowser.insertPlainText("Skip preparing text data. "
                                                "(captions.txt already exists)\n\n")
                QCoreApplication.processEvents()
            if not os.path.exists("features.pkl"):
                QCoreApplication.processEvents()
                self.LogBrowser.insertPlainText(
                    "Extracting features from training dataset was started. It may takes several minutes...\n")
                extracting_features_from_image(
                    "Flickr8k_Dataset/Flicker8k_Dataset")
                self.LogBrowser.insertPlainText(
                    "Extracting features was finished!\n\n")
                QCoreApplication.processEvents()
            else:
                QCoreApplication.processEvents()
                self.LogBrowser.insertPlainText("Skip extracting features. "
                                                "(features.pkl already exists)\n\n")
                QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText(
                "Model fit was started. Please wait!\n")
            start_fit_model(10)
            self.LogBrowser.insertPlainText(
                "Fit was finished. Full log you can see in model.log file.\n\n")
            QCoreApplication.processEvents()

    def lite_training_start(self):
        QMessageBox.about(self, "INFO", "While training takes place, the work area may not be available.")
        if self.checkBox.checkState():
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText(
                "Preparing text data was started! It may takes several minutes...\n")
            prepare_text_data()
            self.LogBrowser.insertPlainText(
                "Preparing text data was finished!\n\n")
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText(
                "Extracting features from training dataset was started. It may takes several minutes...\n")
            extracting_features_from_image(
                "Flickr8k_Dataset/Flicker8k_Dataset")
            self.LogBrowser.insertPlainText(
                "Extracting features was finished!\n\n")
            QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText(
                "Model fit was started. Please wait!\n")
            start_lite_train(10)
            self.LogBrowser.insertPlainText(
                "Fit was finished. Full log you can see in model.log file.\n\n")
            QCoreApplication.processEvents()
        else:
            if not os.path.exists("captions.txt"):
                QCoreApplication.processEvents()
                self.LogBrowser.insertPlainText(
                    "Preparing text data was started! It may takes several minutes...\n")
                prepare_text_data()
                self.LogBrowser.insertPlainText(
                    "Preparing text data was finished!\n\n")
                QCoreApplication.processEvents()
            else:
                self.LogBrowser.insertPlainText("Skip preparing text data. "
                                                "(captions.txt already exists)\n\n")
                QCoreApplication.processEvents()
            if not os.path.exists("features.pkl"):
                self.LogBrowser.insertPlainText(
                    "Extracting features from training dataset was started. It may takes several minutes...\n")
                QApplication.processEvents()
                extracting_features_from_image(
                    "Flickr8k_Dataset/Flicker8k_Dataset")
                self.LogBrowser.insertPlainText(
                    "Extracting features was finished!\n\n")
                QCoreApplication.processEvents()
            else:
                self.LogBrowser.insertPlainText("Skip extracting features. "
                                                "(features.pkl already exists)\n\n")
                QCoreApplication.processEvents()
            self.LogBrowser.insertPlainText(
                "Model lite fit was started. Please wait!\n")
            start_lite_train(10)
            self.LogBrowser.insertPlainText(
                "Lite fit was finished. Full log you can see in model.log file.\n\n")
            QCoreApplication.processEvents()

    def prediction(self):
        filepath = self.imagePathLineEdit.text()
        modelpath = self.modelPathLineEdit_2.text()
        caption = get_predict(filepath, modelpath)
        self.LogBrowser.insertPlainText("Caption for specified image {} with current model {}: \n{}\n\n".format(filepath.split("/")[-1], modelpath.split("/")[-1], caption))


def main():
    self = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    self.exec_()


if __name__ == '__main__':
    main()
