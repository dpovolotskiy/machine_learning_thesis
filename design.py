# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(803, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imagePreView = QtWidgets.QGraphicsView(self.centralwidget)
        self.imagePreView.setGeometry(QtCore.QRect(0, 0, 481, 351))
        self.imagePreView.setObjectName("imagePreView")
        self.LogBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.LogBrowser.setGeometry(QtCore.QRect(0, 360, 801, 211))
        self.LogBrowser.setObjectName("LogBrowser")
        self.tariningPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.tariningPushButton.setGeometry(QtCore.QRect(490, 0, 301, 41))
        self.tariningPushButton.setObjectName("tariningPushButton")
        self.liteTrainingPushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.liteTrainingPushButton_2.setGeometry(QtCore.QRect(490, 40, 301, 41))
        self.liteTrainingPushButton_2.setObjectName("liteTrainingPushButton_2")
        self.predictionButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.predictionButton_3.setGeometry(QtCore.QRect(490, 310, 301, 41))
        self.predictionButton_3.setObjectName("predictionButton_3")
        self.imagePathLineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.imagePathLineEdit.setGeometry(QtCore.QRect(490, 150, 301, 31))
        self.imagePathLineEdit.setReadOnly(True)
        self.imagePathLineEdit.setObjectName("imagePathLineEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(490, 130, 151, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(490, 210, 161, 30))
        self.label_2.setObjectName("label_2")
        self.modelPathLineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.modelPathLineEdit_2.setGeometry(QtCore.QRect(490, 240, 301, 31))
        self.modelPathLineEdit_2.setReadOnly(True)
        self.modelPathLineEdit_2.setObjectName("modelPathLineEdit_2")
        self.pathImageButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pathImageButton_4.setGeometry(QtCore.QRect(490, 180, 301, 31))
        self.pathImageButton_4.setObjectName("pathImageButton_4")
        self.pathModelButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pathModelButton_5.setGeometry(QtCore.QRect(490, 270, 301, 31))
        self.pathModelButton_5.setObjectName("pathModelButton_5")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(490, 80, 131, 21))
        self.checkBox.setObjectName("checkBox")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 803, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tariningPushButton.setText(_translate("MainWindow", "Training (minimal 32 RAM)"))
        self.liteTrainingPushButton_2.setText(_translate("MainWindow", "Lite Training"))
        self.predictionButton_3.setText(_translate("MainWindow", "Prediction"))
        self.label.setText(_translate("MainWindow", "Path to image"))
        self.label_2.setText(_translate("MainWindow", "Path to model"))
        self.pathImageButton_4.setText(_translate("MainWindow", "Choose image path"))
        self.pathModelButton_5.setText(_translate("MainWindow", "Choose model path"))
        self.checkBox.setText(_translate("MainWindow", "Force training"))

