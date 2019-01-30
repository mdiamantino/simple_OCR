__author__ = "Mark Diamantino Caribé"
__copyright__ = "Copyright 2017-2018, Projet de fin d'année - ULB, B1-INFO"
from MPLClass import MPL
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from matplotlib import pyplot as plt
import cv2
import basicUI
import sys
import os


class GuiConnect:
    def __init__(self, gui, application):
        self.progressbarstep = 0
        self.savedocrspinbox = 1
        self.testsetlink, self.imgtoarratlink = '', ''
        self.image, self.picfromarr, self.loadedW1, self.loadedW2, = None, None, None, None
        self.pictoarray, self.imglabel, self.ocrpic, self.uploadedscanpath = None, None, None, None
        self.ex, self.app = gui, application
        self.ex.show()
        self.mpl = MPL(app, self.ex)
        self.manualocrmode = False
        self.ex.anothertestcheck.stateChanged.connect(self.second_dataset)
        self.ex.dodropcheck.stateChanged.connect(self.dropoutwarning)
        self.ex.saveweightscheck.stateChanged.connect(self.checkiftosaveweights)
        self.ex.weigtshloadedcheck.stateChanged.connect(self.useweights)
        self.ex.ocruploadpushbutton.clicked.connect(self.scanupload)
        self.ex.loadwegithsb.clicked.connect(self.loadweightspushed)
        self.ex.ocrmodecombo.currentIndexChanged.connect(self.ocrmodemenu)
        self.ex.startocrpushbutton.clicked.connect(self.ocr)
        self.ex.tabWidget.currentChanged.connect(self.tabchanged)
        self.ex.toolsloadimgpush.clicked.connect(self.uploadimgtoarray)
        self.ex.leftrightconversion.clicked.connect(self.arraytoimg)
        self.ex.rightleftconversion.clicked.connect(self.imgtoarray)
        self.mainwidgets = (
            self.ex.startbutton, self.ex.trainingtestspinbox, self.ex.activationbox, self.ex.lratespinbox,
            self.ex.crossvalspinbox, self.ex.weigtshloadedcheck, self.ex.weightsinspinbox, self.ex.hiddenspinbox,
            self.ex.trainingepochsspin, self.ex.loadwegithsb, self.ex.saveweightscheck, self.ex.dodropcheck,
            self.ex.anothertestcheck)
        self.displayconfusion(True)
        self.ex.startbutton.clicked.connect(self.startpushed)
        self.ex.pausebutton.clicked.connect(self.pausethread)
        sys.exit(self.app.exec())

    def getlink(self):
        """
        :return: (str) Path of the chosen file.
        """
        return QtWidgets.QFileDialog.getOpenFileName(self.ex, 'Open File')[0]

    @staticmethod
    def pausethread():
        """
        Pause the event running when this function is called.
        """
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("PAUSE")
        msg.setInformativeText("Click on 'OK' in order to resume the Cross-Validation :D")
        msg.setDefaultButton(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def tabchanged(self):
        """
        If the user is on tab 'tools', most of upper cross-validation widgets are disabled being unnecessary.
        """
        temp = self.mainwidgets[:8]  # Widgets to temporarily disable.
        if self.ex.tabWidget.currentIndex() == 2:
            for elem in temp:
                elem.setEnabled(False)
        else:
            for elem in temp:
                elem.setEnabled(True)

    def displayconfusion(self, initialization):
        """
        Displays the confusion matrix.
        :param initialization: Boolean ( if True, displays en empty plot )
        """
        if initialization:
            self.image = self.plot_confusionmatrix(np.zeros((10, 10)), True).copy()
        else:
            self.image = self.plot_confusionmatrix(self.mpl.confusion, False).copy()
        self.imglabel = QtWidgets.QLabel(self.ex)
        self.imglabel = self.convertopyqtimage(self.image, self.imglabel)
        self.ex.gridLayout_3.addWidget(self.imglabel, 2, 1, 1, 1)

    def plot_confusionmatrix(self, conf, empty=False):
        """
        Plots a numpy array and converts it to an Open-cv image.
        :param conf: Confusion matrix
        :param empty: Boolean ( if True, initializes an empty plot )
        :return: Open-cv confusion matrix plot image
        """
        new = conf
        if not empty:
            new = np.array([np.round(np.divide(line, np.sum(line)), 3) for line in conf])
        base = plt.figure(figsize=(9, 9))
        edit = base.add_subplot(111)
        mappable = edit.imshow(np.array(new), interpolation='nearest', cmap=plt.get_cmap('Blues'))
        if not empty:
            for x in range(10):
                for y in range(10):
                    edit.annotate(str(new[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')
        plt.xticks(np.arange(10), np.arange(10)), plt.yticks(np.arange(10), np.arange(10))
        plt.xlabel('Predicted Labels'), plt.ylabel('Real Labels')
        plt.colorbar(mappable)
        img = self.matpltlib_to_opencv(base)
        return img[100:0 + 820, 50:0 + 820]

    def clearandsiplaytext(self, text):
        """
        Displays a string in the text browser of the 'Tools' tab.
        :param text: (str) that has to be displayed in the 'Tools' tab inside the text browser.
        """
        self.ex.arraybrows.clear()
        self.ex.arraybrows.appendPlainText(text)
        self.ex.arraybrows.update()

    def uploadimgtoarray(self):
        """
        Lets the user load and display the selected image in order to convert it to an array.
        In case the image aspect ratio is too different from the most favorable one ( 1:1 ),
        it is asked to load another picture.
        """
        temp = self.getlink()
        pic = cv2.imread(temp)
        height, width, channels = pic.shape
        if min(height, width) / max(height, width) < 0.8:
            limitsexceeded = 'The limits of the allowed aspect ratio of images are 5:4 or 4:5.' \
                             '\nTry to find an image with an aspect ratio as close as possible to 1:1 ' \
                             'in order to perform a better conversion to the MNIST format.'
            self.clearandsiplaytext(limitsexceeded)
        else:
            self.imgtoarratlink = temp
            self.ex.toolsloadimgpush.setText("Loaded")
            self.ex.toolsloadimgpush.setEnabled(False)
            self.ex.rightleftconversion.setEnabled(True)
            self.pictoarray = QtWidgets.QLabel(self.ex)
            self.pictoarray = self.convertopyqtimage(self.resize_opencv(pic, 600), self.pictoarray)
            self.ex.conversiongridLayout.addWidget(self.pictoarray, 0, 0)

    def useweights(self):
        """
        If W1 and W2 have been loaded and the user check the 'use them' checkbox,
        the training widgets are disabled and the title of the ocr start button is changed,
        in order to let the user understand that the OCR will use them without (re)starting a training.
        """
        tempdico = {True: "Start OCR (Will use loaded weights)", False: "Start OCR (Will train first to have weights)"}
        # Changes the OCR start button title.
        self.ex.startocrpushbutton.setText(tempdico[self.ex.weigtshloadedcheck.isChecked()])
        for wget in (self.ex.trainingepochsspin, self.ex.hiddenspinbox, self.ex.lratespinbox, self.ex.weightsinspinbox):
            wget.setDisabled(self.ex.weigtshloadedcheck.isChecked())

    def scantoarrays(self, path, n):
        """
        Converts the scanned image to multiple resized, reshaped, framed digit arrays.
        :param path: (str) Path of the selected image for the OCR mode.
        :param n: (int) Number of digits in the selected image (helps improve the handwritten digits recognition)
        :return: (matrix n*784) Sorted matrix where each line is a digit (MNIST format) of the loaded image.
        """
        digits, helpsort, areas = [], [], []  # To help sort the recognised digits according to their position in the image.
        imc = (cv2.imread(path)).copy()
        greyblur = cv2.blur(cv2.cvtColor(imc, cv2.COLOR_BGR2GRAY), (5, 5))  # Greyscale and blur filter
        (thresh, greyblur2) = cv2.threshold(greyblur, 200, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # Threshold filter
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))  # Rounds shapes
        thresh = cv2.morphologyEx(greyblur2, cv2.MORPH_OPEN, kernel)  # Rounds shapes
        imc2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if self.manualocrmode:
            areas = sorted([cv2.contourArea(k) for k in contours])  # Sorts areas of found contours.
            print("Manual Mode")
        else:
            areas = [cv2.contourArea(k) for k in contours if cv2.contourArea(k) > 500]
        for (i, c) in enumerate(contours):
            #if cv2.contourArea(c) in areas[-n:]:  # Takes the n biggest areas (usually there are all the digits in).
            if cv2.contourArea(c) in (areas[-n:] if self.manualocrmode else areas):
                [x, y, w, h] = cv2.boundingRect(c)
                helpsort.append(x)
                cv2.rectangle(imc, (x, y), (x + w, y + h), (0, 0, 255), 7)  # Draw a rectange on it.
                add = int(max(h, w) * 1.3) - max(h, w)  # In order to make a perfect square around the digit
                if h > w:  # In order to have 20:28 proportions (digit/area)
                    squarew = (h - w) // 2 + add
                    squereh = 0 + add
                else:
                    squarew = 0 + add
                    squereh = (w - h) // 2 + add
                roi = thresh[y - squereh:y + h + squereh, x - squarew:x + w + squarew]
                #cv2.imshow('test', roi)
                digit = cv2.resize(roi, (28, 28))  # Digits are resized to hte standard MNIST format.
                digits.append(self.cv2array(digit))
                #key = cv2.waitKey(0)
        self.displayocr(imc)  # Found digits are displayed.
        return [x for _, x in sorted(zip(helpsort, digits))]  # They are sorted according to their initial position.

    @staticmethod
    def resize_opencv(picture, maxsize):
        """
        If the biggest size of a picture exceeds the allowed size,
        the picture is resized to the a better one ( keeping proportions ).
        :param picture: Open-cv image
        :param maxsize: (int) Allowed max size
        :return: Open-cv resized picture if exceeds, else the original one.
        """
        res = picture
        height, width, channel = picture.shape
        if max(height, width) > maxsize:
            if height > width:
                temp = maxsize / height
            else:
                temp = maxsize / width
            res = cv2.resize(picture, (0, 0), fx=temp, fy=temp)
        return res

    def displayocr(self, ocrpic):
        """
        Displays the loaded picture in the OCR tab.
        :param ocrpic: Open-cv Image
        """
        self.ocrpic = QtWidgets.QLabel(self.ex)
        self.ocrpic = self.convertopyqtimage(self.resize_opencv(ocrpic, 1100), self.ocrpic)
        self.ex.qrgridLayout.addWidget(self.ocrpic, 1, 0)

    def ocrmodemenu(self):
        """
        In case the user chooses the manual mode.
        """
        if self.ex.ocrmodecombo.currentIndex() == 1:
            self.ex.ocrdigitspinbox.setValue(self.savedocrspinbox)  # Restores the saved value of the digits spinbox.
            self.ex.ocrdigitspinbox.setEnabled(True)
            self.manualocrmode = True
        else:   # Auto Mode enabled
            self.savedocrspinbox = self.ex.ocrdigitspinbox.value()  # Saves the current value of the digits spinbox.
            self.ex.ocrdigitspinbox.setValue(1)  # Sets the digits spinbox to 1 and does not let the user change it.
            self.ex.ocrdigitspinbox.setEnabled(False)  # Allows the user to change the value of the digits spinbox.
            self.manualocrmode = False

    def scanupload(self):
        """
        In case the user wants to load the OCR image.
        """
        if self.ex.startocrpushbutton.isEnabled():  # If user can upload (when no cross-validation is running).
            self.uploadedscanpath = self.getlink()
            self.ex.ocruploadpushbutton.setText("Loaded")
            self.ex.ocruploadpushbutton.setDisabled(True)  # Can't upload another picture until finished.

    def loadweightspushed(self):
        """
        In case the user wants to load pre-saved or not present weights.
        """
        if os.path.isfile('W1.csv') and os.path.isfile('W2.csv'):  # If weights are in the same directory.
            self.loadedW1, self.loadedW2 = self.read_weights()
            self.ex.weigtshloadedcheck.setCheckable(True)
            self.ex.loadwegithsb.setText("Loaded")
            self.ex.loadwegithsb.setEnabled(False)
            self.ex.saveweightscheck.setChecked(False)
            self.ex.saveweightscheck.setEnabled(False)
        else:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Weights information")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("'W1.csv' and 'W2.csv' were not found in this directory. ")
            msg.setInformativeText("There is additional information...")
            msg.setDetailedText(
                "If you do not have them yet, click on 'Save weights', start the Cross-Validation and wait its end. Then you will be able to use them. If you already have them, put them in the same directory of the main code.")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()

    @staticmethod
    def cv2array(picture):
        """
        Converts an Open-cv image to a numpy array.
        :param picture: Open-cv image
        :return: Numpy array of the given open-cv image.
        """
        img = Image.fromarray(picture)
        return np.array(img.getdata())

    def ocr(self):
        self.mpl.act = 2
        W1ocr, W2ocr = self.loadedW1, self.loadedW2
        if not ex.weigtshloadedcheck.isChecked():
            self.updatemplvalues(True)
            W1ocr, W2ocr = self.mpl.trainperceptron()
        ocresult = ''
        #if self.manualocrmode:
        #    ocresult += str(self.mpl.predict(self.image_to_mnist(self.uploadedscanpath), W1ocr, W2ocr))
        #    self.displayocr(cv2.imread(self.uploadedscanpath))
        for picture in self.scantoarrays(self.uploadedscanpath, self.ex.ocrdigitspinbox.value()):
            ocresult += str(self.mpl.predict(picture, W1ocr, W2ocr)) + ' '
        self.ex.digitsequence.setText(ocresult)
        self.ex.ocruploadpushbutton.setText("Load Image")
        self.ex.ocruploadpushbutton.setEnabled(True)

    @staticmethod
    def read_data(datapath):
        """
        Reads and convert the path of a given .csv file to a usable np array format dataset.
        :param datapath: (str) Path of the dataset
        :return: (np. array) of the converted dataset.
        """
        with open(datapath) as f:
            return np.array([[int(s) for s in line.split(',')] for line in f])

    @staticmethod
    def matpltlib_to_opencv(figure):
        """
        Converts from matplotlib to open-cv.
        :param figure: Matplotlib figure
        :return: Open-cv figure
        """
        figure.canvas.draw()
        img = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def convertopyqtimage(pic, lab):
        """
        Converts an Open-CV to a PyQt5 label-image.
        :param pic:
        :param lab: Pyqt label to replace with the desired image.
        :return: Edited PyQt5 label (now containing the image)
        """
        height, width, channel = pic.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(pic.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        qImg = qImg.rgbSwapped()
        lab.setPixmap(QtGui.QPixmap.fromImage(qImg))
        lab.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        return lab

    def arraytoimg(self):
        """
        Main conversion function of the 'Tools' tab. Converts the typed array to image and displays it.
        """
        errornum = False
        text = self.ex.arraybrows.toPlainText()  # What the user has written
        if len(text) == 0:
            self.clearandsiplaytext('Please type something ( possibly an array ) and try again.')
        else:
            try:
                arr = eval(text)
                lenis784 = True if len(arr) == 784 else False
                for number in arr:
                    if number < 0 or number > 255:
                        errornum = True
                if lenis784 and not errornum:
                    arr = np.array(arr, dtype='uint8')
                    arr = arr.reshape((28, 28))
                    picturefromarray = cv2.resize(cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR), (600, 600))
                    self.picfromarr = QtWidgets.QLabel(self.ex)
                    self.picfromarr = self.convertopyqtimage(picturefromarray, self.picfromarr)
                    self.ex.conversiongridLayout.addWidget(self.picfromarr, 0, 0)
                else:
                    phrase = '\nPlease fix it and try again.'
                    if errornum:
                        self.clearandsiplaytext('The array contains a value less than 0 or greater than 255.' + phrase)
                    elif not lenis784:
                        self.clearandsiplaytext('The number of elements in the array is different from 784.' + phrase)
            except TypeError:
                self.clearandsiplaytext('The type of the entered value is wrong.\nPlease re-write it and try again.')
            except SyntaxError:
                self.clearandsiplaytext('The syntax of the typed array is wrong.\nPlease re-write it and try again.')

    def imgtoarray(self):
        """
        Displays the loaded image as array in the 'Tools' tab.
        """
        arr = self.image_to_mnist(self.imgtoarratlink)
        self.ex.arraybrows.clear()
        rep = np.array2string(arr, precision=1, separator=',', suppress_small=True)
        self.ex.arraybrows.appendPlainText(rep)
        self.ex.arraybrows.update()

    @staticmethod
    def image_to_mnist(picturepath):
        """
        Image to MNIST format converter.
        :param picturepath: Path of the selected image.
        :return: np. array mnist format of the selected image.
        """
        img = Image.open(picturepath)
        s = min(img.size)
        left = (img.size[0] - s) / 2
        top = (img.size[1] - s) / 2
        right = (img.size[0] + s) / 2
        bottom = (img.size[1] + s) / 2
        cropped = img.crop((left, top, right, bottom))
        reshaped = cropped.resize((28, 28), Image.ANTIALIAS)
        filtered = reshaped.filter(ImageFilter.SHARPEN)
        inverted = ImageOps.invert(filtered.convert('L'))
        return np.array(inverted.getdata())

    def save_weights(self):
        """
        Saves the weights in two csv files in the same directory.
        """
        W1, W2 = self.mpl.sendweights
        np.savetxt("W1.csv", W1, delimiter=",")
        np.savetxt("W2.csv", W2, delimiter=",")

    @staticmethod
    def read_weights():
        """
        Reads the two csv weights file (in the same directory) and converts them to exploitable np. array weights.
        :return: Two np. array weights.
        """
        return np.genfromtxt('W1.csv', 'float', delimiter=','), np.genfromtxt('W2.csv', 'float', delimiter=',')

    def second_dataset(self):
        """
        In case the user wants to use another dataset as testset.
        """
        self.ex.testsetbox.setEnabled(self.ex.anothertestcheck.isChecked())
        self.ex.trainingtestspinbox.setDisabled(self.ex.anothertestcheck.isChecked())

    def checkiftosaveweights(self):
        """
        In case the user wants to save weights.
        """
        self.ex.loadwegithsb.setDisabled(self.ex.saveweightscheck.isChecked())

    def dropoutwarning(self):
        """
        Shows a dropout warning, because dropout can lower scores.
        """
        if ex.dodropcheck.isChecked():
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Dropout information")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText(
                "Dropout could significantly decrease the gobal score. We strongly suggest you to leave it off.")
            msg.setInformativeText("There is additional information...")
            msg.setDetailedText(
                "Even if the probability is kept low, in this case the used dropout method is not fully compatible with this machine learning structure. ")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ignore | QtWidgets.QMessageBox.Cancel)
            result = msg.exec_()
            if result == QtWidgets.QMessageBox.Ignore:
                self.ex.dropurcentagespinbox.setEnabled(True)
            elif result == QtWidgets.QMessageBox.Cancel:
                self.ex.dodropcheck.setChecked(False)
                self.ex.dropurcentagespinbox.setEnabled(False)
        else:
            self.ex.dropurcentagespinbox.setEnabled(False)
            self.ex.dodropcheck.setChecked(False)

    def updatemplvalues(self, forocr):
        """
        Updates mpl values.
        :param forocr: (boolean) If True, some values are changed.
        """
        self.mpl.data, self.mpl.testset, self.mpl.act, self.mpl.ninputs, self.mpl.nhidden, self.mpl.noutput, self.mpl.lrate, self.mpl.dweight, self.mpl.crossit, self.mpl.epochs, self.mpl.dodropout, self.mpl.dropout_percent = self.getupvalues()
        if forocr:
            self.ex.crossvalspinbox.setValue(1)
            self.mpl.crossit = 1
        self.mpl.singlestep = 100 / (ex.trainingepochsspin.value() * ex.crossvalspinbox.value())
        self.mpl.shouldisave = self.ex.saveweightscheck.isChecked()
        self.mpl.useloaded = self.ex.weigtshloadedcheck.isChecked()
        self.mpl.loadedW1, self.mpl.loadedW2 = self.loadedW1, self.loadedW2

    def getupvalues(self):
        """
        Gets the values of the up widgets.
        :return: Values of the up widgets
        """
        traindico = {0: 'train.csv', 1: 'train_small.csv', 2: 'train_tiny.csv'}
        testdico = {2: 'train.csv', 0: 'train_small.csv', 1: 'train_tiny.csv'}
        if self.ex.anothertestcheck.isChecked():
            trainingset = self.read_data(traindico[self.ex.trainingsetbox.currentIndex()])
            if self.ex.testsetbox.currentIndex() != 3:
                testset = self.read_data(testdico[self.ex.testsetbox.currentIndex()])
            else:
                testset = self.read_data(self.testsetlink)
        else:
            dataset = self.read_data(traindico[self.ex.trainingsetbox.currentIndex()])
            training_size = int(self.ex.trainingtestspinbox.value() * len(dataset))
            trainingset = dataset[:training_size]
            testset = dataset[training_size:]
        activationf = self.ex.activationbox.currentIndex()
        hidden_size = self.ex.hiddenspinbox.value()
        lrate = self.ex.lratespinbox.value()
        trainingit = self.ex.trainingepochsspin.value()
        cvit = self.ex.crossvalspinbox.value()
        dropprob = self.ex.dropurcentagespinbox.value()
        dodrop = self.ex.dodropcheck.isChecked()
        dweight = self.ex.weightsinspinbox.value()
        return trainingset, testset, activationf, 784, hidden_size, 10, lrate, dweight, cvit, trainingit, dodrop, dropprob

    def startpushed(self):
        """
        In case the start button is pushed.
        """
        tempenabled = [elem for elem in self.mainwidgets if elem.isEnabled()]  # Save the value of enabled widgets.
        self.ex.startbutton.setEnabled(False)
        self.ex.pausebutton.setEnabled(True)
        self.ex.progressBar.setValue(0)
        self.mpl.logs = ''
        self.ex.clearterminal()
        self.app.processEvents()
        self.progressbarstep = 100 / (self.ex.trainingepochsspin.value() * self.ex.crossvalspinbox.value())
        self.updatemplvalues(False)
        for elem in tempenabled:  # Disables saved widgets.
            elem.setEnabled(False)
        self.mpl.crossValidation()
        for elem in tempenabled:  # Re-activates widgets.
            elem.setEnabled(True)
        self.displayconfusion(False)
        if self.ex.saveweightscheck.isChecked():
            self.save_weights()
        if self.ex.savelogscheck.isChecked():
            with open("Logs.txt", "w") as lg:
                lg.write(self.mpl.logs)
        self.ex.startbutton.setEnabled(True)
        self.ex.pausebutton.setEnabled(False)
        self.ex.loadwegithsb.setText("Load Weights")
        self.ex.loadwegithsb.setEnabled(True)
        self.ex.weigtshloadedcheck.setChecked(False)
        self.ex.weigtshloadedcheck.setCheckable(False)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = basicUI.Ui_Dialog()
    main = GuiConnect(ex, app)
