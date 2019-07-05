import os
import sys
import time
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import cv2
from ui_main2 import *
from FaceFeatureExtractor import *
from Prediction import *
from PickleGenerator import *
from ClassifierClass import *
from dataset_creator import *
from FeatureExtractorLBPH import *
from FeatureExtractorLMDeP import *
from AnalysisGraph import *

class MainWindow(QtWidgets.QMainWindow):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.name = ""
        self._min_size = (180, 180)
        self.img = []
        self.rimg = []
        self.rname = ""
        self.i = 0
        self.imgname = []
        self.imgsrc = []
        self.imageinadex = 0
        self.previousTab = 0

        self.st_precision = 0
        self.st_recall = 0
        self.st_accuracy = 0
        self.st_time = 0
        self.st_maxdatasetvalue = 200
        self.st_featuretype = 1
        self.st_slidervalue = 70

        self.ui.tabWidget.setCurrentIndex(0)

        # load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier('HAARFace.xml')
        if self.face_cascade.empty():
            QMessageBox.information(self , "Error Loading cascade classifier", "Unable to load face cascade file")
            sys.exit()

        #Add Page
        self.timer = QTimer()
        self.timer.timeout.connect(self.detectFaces)
        self.ui.start_bt.clicked.connect(self.controlTimer)
        self.ui.capture_bt.clicked.connect(self.captureFaces)
        self.ui.nextImage_bt.clicked.connect(self.nextImage)
        self.ui.previousImage_bt.clicked.connect(self.prevoiusImage)
        self.ui.name_text.returnPressed.connect(self.nameChnagetevent)
        self.ui.process_bt.clicked.connect(self.processImages)
        self.ui.reset_bt.clicked.connect(self.resetAddPage)
        self.ui.next_dataset_bt.clicked.connect(self.nextDatabase_Bt_Pressed)
        self.ui.createpkl_bt.clicked.connect(self.createpicklefun)
        self.ui.tabWidget.currentChanged.connect(self.changeTab)
        self.ui.createpkl_bt.setEnabled(False)


        #Recognise Page
        self.rtimer = QTimer()
        self.rtimer.timeout.connect(self.rdetectFaces)
        self.ui.rstart_bt.clicked.connect(self.rcontrolTimer)
        self.ui.rrecognise_bt.clicked.connect(self.rrecogniseFace)
        self.ui.rreset_bt.clicked.connect(self.rresetRecognisePage)


        #Standard Page
        self.ui.st_process_bt.clicked.connect(self.st_process)
        self.ui.st_horizontalSlider.valueChanged.connect(self.st_slidervaluechange)
        self.ui.st_slidervalue_lineEdit.editingFinished.connect(self.st_sliderlineeditchange)
        self.onlyInt = QtGui.QIntValidator()
        self.ui.st_slidervalue_lineEdit.setValidator(self.onlyInt)

        #Analysis Page
        self.ui.an_process_lbph_bt.clicked.connect(self.an_process_lbph)
        self.ui.an_process_lmdep_bt.clicked.connect(self.an_process_lmdep)

    def st_sliderlineeditchange(self):
        i = int(self.ui.st_slidervalue_lineEdit.text())
        self.ui.st_horizontalSlider.setValue(i)
        self.st_slidervalue = i
    def st_slidervaluechange(self):
        i = self.ui.st_horizontalSlider.value()
        j = i%10
        i-=j
        if j>5:
            i = i+10
        self.ui.st_slidervalue_lineEdit.setText(str(i))
        self.ui.st_horizontalSlider.setValue(i)
        self.i = i

    def an_process_lbph(self):
        nds1 = pd.read_csv("LBPH Features/FaceFeaturesFace94LBPH.csv")
        nds2 = pd.read_csv("LBPH Features/FaceFeaturesFace95LBPH.csv")
        nds3 = pd.read_csv("LBPH Features/FaceFeaturesFace96LBPH.csv")
        nds4 = pd.read_csv("LBPH Features/FaceFeaturesFEIFaceLBPH.csv")
        dataanalysis = DataAnalysis(nds1,nds2,nds3,nds4)
        result = dataanalysis.drawGraph()
        frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show frame in img_label
        self.ui.an_LBPH_label.setPixmap(QPixmap.fromImage(qImg))
        self.ui.an_LBPH_label.setAlignment(QtCore.Qt.AlignCenter)

    def an_process_lmdep(self):
        nds1 = pd.read_csv("LMDEP Features/FaceFeaturesFace94LMDEP.csv")
        nds2 = pd.read_csv("LMDEP Features/FaceFeaturesFace95LMDEP.csv")
        nds3 = pd.read_csv("LMDEP Features/FaceFeaturesFace96LMDep.csv")
        nds4 = pd.read_csv("LMDEP Features/FaceFeaturesFEIFaceLMDep.csv")
        dataanalysis2 = DataAnalysis(nds1,nds2,nds3,nds4)
        result = dataanalysis2.drawGraph()
        frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show frame in img_label
        self.ui.an_LMDEP_label.setPixmap(QPixmap.fromImage(qImg))
        self.ui.an_LMDEP_label.setAlignment(QtCore.Qt.AlignCenter)

    def st_process(self):
        tstart = time.process_time()
        if self.ui.st_radioButton_d1.isChecked():
            d = Dataset_creator("Database1")
            self.imgsrc, self.imgname = d.create_dataset()
        elif self.ui.st_radioButton_d2.isChecked():
            d = Dataset_creator("Database2")
            self.imgsrc, self.imgname = d.create_dataset()
        elif self.ui.st_radioButton_d3.isChecked():
            d = Dataset_creator("Database3")
            self.imgsrc, self.imgname = d.create_dataset()
        elif self.ui.st_radioButton_d4.isChecked():
            d = Dataset_creator("Database4")
            self.imgsrc, self.imgname = d.create_dataset()

        self.ui.progressBar.setValue(10)

        if self.ui.st_radioButton_LBPH.isChecked():
            print(self.imgsrc)
            print(self.imgname)
            self.st_featuretype = 1
            lbph = LocalBinaryPatterns(self.imgsrc,self.imgname,self.ui.st_maxdata_spinbox.value())
            csvpath = lbph.featureExtractLbph()
            print(csvpath)
        elif self.ui.st_radioButton_LMDEP.isChecked():
            print(self.imgsrc)
            print(self.imgname)
            self.st_featuretype = 2
            lmdep = LMDeP(self.imgsrc,self.imgname,200)
            csvpath = lmdep.featureExtractLmdep()

        self.ui.progressBar.setValue(50)
        nds = pd.read_csv("FaceFeatures.csv")

        if self.ui.st_radioButton_RF.isChecked():
            print("RF")
            cf = Classifier(nds, self.st_featuretype, self.st_slidervalue)
            print ("qwerty")
            self.st_precision,self.st_recall,self.st_accuracy= cf.randomForest()
        elif self.ui.st_radioButton_ED.isChecked():
            print("ED")
            cf = Classifier(nds, self.st_featuretype, self.st_slidervalue)
            print ("qwerty")
            self.st_precision, self.st_recall, self.st_accuracy= cf.euclidianDistance()
        elif self.ui.st_radioButton_SVM.isChecked():
            print("SVM")
            cf = Classifier(nds, self.st_featuretype, self.st_slidervalue)
            print ("qwerty")
            self.st_precision, self.st_recall, self.st_accuracy= cf.supportVectormachine()
        elif self.ui.st_radioButton_KNN.isChecked():
            print("KNN+")
            cf = Classifier(nds, self.st_featuretype, self.st_slidervalue)
            print ("qwerty")
            self.st_precision, self.st_recall, self.st_accuracy= cf.kNearestNeighbour()

        self.ui.st_time_result.setText(str(time.process_time() - tstart))
        self.ui.progressBar.setValue(100)
        self.ui.st_precision_result.setText(str(self.st_precision))
        self.ui.st_recall_result.setText(str(self.st_recall))
        self.ui.st_accuracy_result.setText(str(self.st_accuracy))

    def changeTab(self):
        print(self.previousTab)
        if self.previousTab==0:
            if self.timer.isActive():
                self.timer.stop()
                self.cap.release()
            self.resetAddPage()

        elif self.previousTab==1:
                if self.rtimer.isActive():
                    self.rtimer.stop()
                    self.rcap.release()
                self.rresetRecognisePage()

        self.previousTab = self.ui.tabWidget.currentIndex()

    def nextDatabase_Bt_Pressed(self):
        self.resetAddPage()
        self.ui.reset_bt.setEnabled(True)

    def resetAddPage(self):
        self.name = ""
        self._min_size = (180, 180)
        self.img = []
        self.i = 0
        self.imageinadex = 0
        self.ui.name_text.setText("")
        self.ui.name_text.setEnabled(True)
        self.ui.start_bt.setEnabled(False)
        self.ui.capture_bt.setEnabled(False)
        self.ui.process_bt.setEnabled(False)
        self.ui.next_dataset_bt.setEnabled(False)
        self.ui.lcdNumber.display(self.i)
        self.ui.image_label.setText("")
        self.ui.camera_label.setText("")
        if self.timer.isActive():
            self.timer.stop()
            self.cap.release()
        self.ui.start_bt.setText("Start")
        self.ui.image_label.clear()
        self.ui.camera_label.clear()

    def createpicklefun(self):
        pickleobject = PickleGenerator('faceFeature.csv')
        pickleobject.pklCreate()

    def processImages(self):
        self.ui.reset_bt.setEnabled(False)

        for j in range(self.i):
            directory = "Face/" + str(self.name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            k= str(directory)+"/" + str(j) + ".jpg"
            FindFace(cv2.imread(k), self.name.replace(" ", "_"))
        self.ui.next_dataset_bt.setEnabled(True)
        self.ui.process_bt.setEnabled(False)

    def nameChnagetevent(self):
        self.ui.start_bt.setEnabled(True)
        self.ui.name_text.setEnabled(False)

    def captureFaces(self):
        if self.i < 20 and self.i >= 0:
            directory= "Face/" + str(self.name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            k= str(directory)+"/" + str(self.i) + ".jpg"
            self.imageinadex=self.i
            try:cv2.imwrite(k,self.img)
            except :
                self.ui.statusbar.showMessage("Image Not Detected", 1000)
                return
            frame = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            # show frame in img_label
            self.ui.lcdNumber.display(self.i)
            self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
            self.ui.image_label.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.process_bt.setEnabled(True)
            self.i+=1
            self.img=[]
        else:
            self.ui.capture_bt.setEnabled(False)

    def nextImage(self):
        directory = "Face/" + str(self.name)
        k1 = str(directory)+"/" + str(self.imageinadex+1) + ".jpg"
        if os.path.isfile(k1):
            self.imageinadex += 1
            img = cv2.imread(k1)
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            # show frame in img_label
            self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
            self.ui.image_label.setAlignment(QtCore.Qt.AlignCenter)

    def prevoiusImage(self):
        directory = "Face/" + str(self.name)
        k1 = str(directory)+"/" + str(self.imageinadex-1) + ".jpg"
        if os.path.isfile(k1):
            self.imageinadex = self.imageinadex - 1
            img = cv2.imread(k1)
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            # show frame in img_label
            self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
            self.ui.image_label.setAlignment(QtCore.Qt.AlignCenter)

    def detectFaces(self):
        self.name = self.ui.name_text.text()
        # read frame from video capture
        ret, frame = self.cap.read()

        # resize frame image
        scaling_factor = 0.8
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # convert frame to GRAY format
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect rect faces
        face_rects = self.face_cascade.detectMultiScale(gray, 1.3, 4, cv2.CASCADE_SCALE_IMAGE, self._min_size)

        # for all detected faces
        for (x, y, w, h) in face_rects:
            # draw green rect on face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_image = frame[y:y + h, x:x + w]
            self.img = face_image

        # convert frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get frame infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from RGB frame
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show frame in img_label
        self.ui.camera_label.setPixmap(QPixmap.fromImage(qImg))
        self.ui.camera_label.setAlignment(QtCore.Qt.AlignCenter)

    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.capture_bt.setEnabled(True)
            self.ui.start_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.start_bt.setText("Start")

    def rresetRecognisePage(self):
        self.rname= ""
        self.ui.rname_label.setText(self.rname)
        self.ui.rcamera_label.setText(" ")

    def rrecogniseFace(self):
        predict = PredictFace(self.rimg, "faceFeature.csv")
        self.rname = predict.predict()
        self.ui.rname_label.setText(str(self.rname))

    def rdetectFaces(self):
        self.rname = self.ui.name_text.text()
        # read frame from video capture
        ret, frame = self.rcap.read()

        # resize frame image
        scaling_factor = 0.8
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # convert frame to GRAY format
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect rect faces
        face_rects = self.face_cascade.detectMultiScale(gray, 1.3, 4, cv2.CASCADE_SCALE_IMAGE, self._min_size)

        # for all detected faces
        for (x, y, w, h) in face_rects:
            # draw green rect on face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_image = frame[y:y + h, x:x + w]
            self.rimg = face_image

        # convert frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get frame infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from RGB frame
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show frame in img_label
        self.ui.rcamera_label.setPixmap(QPixmap.fromImage(qImg))
        self.ui.rcamera_label.setAlignment(QtCore.Qt.AlignCenter)

    def rcontrolTimer(self):
        # if timer is stopped
        if not self.rtimer.isActive():
            # create video capture
            self.rcap = cv2.VideoCapture(0)
            # start timer
            self.rtimer.start(20)
            # update control_bt text
            self.ui.rrecognise_bt.setEnabled(True)
            self.ui.rstart_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.rtimer.stop()
            # release video capture
            self.rcap.release()
            # update control_bt text
            self.ui.rstart_bt.setText("Start")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())