__author__ = "Mark Diamantino Caribé"
__copyright__ = "Copyright 2017-2018, Projet de fin d'année - ULB, B1-INFO"
import numpy as np
import time


class MPL:
    def __init__(self, guiapp, gui):
        self.guiapp = guiapp
        self.gui = gui
        self.act = 0
        self.ninputs = 784
        self.nhidden = 50
        self.noutput = 10
        self.lrate = 0.0001
        self.dweight = 0.0001
        self.crossit = 1
        self.epochs = 50
        self.dodropout = False
        self.dropout_percent = 0.2
        self.singlestep = 1
        self.shouldisave, self.useloaded, self.processrunning = False, False, False
        self.loadedW1, self.loadedW2, self.data, self.testset = None, None, None, None
        self.logs, self.textoutput = '', ''
        self.activations = [self.softsign, self.arctan, self.tanh, self.logistic]
        self.W1 = np.random.normal(size=(self.ninputs, self.nhidden), scale=self.dweight)
        self.W2 = np.random.normal(size=(self.nhidden, self.noutput), scale=self.dweight)
        self.confusion = np.zeros((10, 10))
        self.progbarpurcentage = 0
        self.sendweights = ()

    @staticmethod
    def logistic(x, derivative=False):
        """
        :param x: Vector, Matrix, Int, Float. (possibily a vector)
        :param derivative: Boolean
        :return: Activated 'x'.
        """
        try:
            output = 1.0 / (1.0 + np.exp(-x))
        except OverflowError:
            output = 0.0000000000000001 if x < 0 else 0.9999999999999999
        return (output if not derivative else (output * (1 - output)))

    @staticmethod
    def tanh(x, derivative=False):
        """
        :param x: Vector, Matrix, Int, Float. (possibily a vector)
        :param derivative: Boolean
        :return: Activated 'x'.
        """
        res = np.tanh(x)
        if derivative:
            res = 1 - res ** 2
        return res

    @staticmethod
    def softsign(x, derivative=False):
        """
        :param x: Vector, Matrix, Int, Float. (possibily a vector)
        :param derivative: Boolean
        :return: Activated 'x'.
        """
        if derivative:
            res = 1 / ((1 + np.abs(x)) ** 2)
        else:
            res = x / (1 + np.abs(x))
        return res

    @staticmethod
    def arctan(x, derivative=False):
        """
        :param x: Vector, Matrix, Int, Float. (possibily a vector)
        :param derivative: Boolean
        :return: Activated 'x'.
        """
        if derivative:
            res = np.divide(1, np.square(x) + 1)
        else:
            res = np.arctan(x)
        return res

    def forwpass(self, image, W1, W2):
        """
        Implementation of forward pass for a multilayer multiclass perceptron.
        :param image: Input image. (vector)
        :param W1: Weight of the first layer. (matrix)
        :param W2: Weight of the last layer. (matrix)
        :return: Input layer, h1: activated first layer, h2: activated second layer
        """
        h0 = image
        a1 = np.dot(h0, W1)
        h1 = self.activations[self.act](a1)
        a2 = np.dot(h1, W2)
        h2 = self.activations[self.act](a2)
        return h0, a1, h1, a2, h2

    def dropout(self):
        """
        Apply a dropout mask to W1, W2.
        """
        self.W1 *= np.random.binomial(1, 1 - self.dropout_percent, self.W1.shape)
        self.W2 *= np.random.binomial(1, 1 - self.dropout_percent, self.W2.shape)

    def backprop(self, label, h0, a1, h1, a2, h2, W2):
        """
        Backpropagation implementation.
        :param label: Real digit label.
        :param h0: Input layer
        :param a1: Work var.
        :param h1: Activated first layer
        :param a2: Work var.
        :param h2: Activated second layer
        :param W2: Weight of the last layer. (matrix)
        :return:
        """
        g = h2 - label
        g = np.array([np.multiply(g, self.activations[self.act](a2, True))])
        DeltaW2 = np.dot(np.array([h1]).T, g)
        g = np.dot(g, W2.T)
        g = np.multiply(g, self.activations[self.act](a1, True))
        DeltaW1 = np.outer(np.array([h0]).T, g)
        return DeltaW1, DeltaW2

    def trainperceptron(self):
        """
        Full training of the multilayer, multiclass perceptron.
        :return: Weights of the first and last layers after training. (matrix)
        """
        lrate_temp = self.lrate
        for epochs in range(self.epochs):
            np.random.shuffle(self.data)
            XTrain = self.data[:, 1:]
            LTrain = self.data[:, 0]
            LTrain = np.array([[1 if j == LTrain[i] else 0 for j in range(self.noutput)] for i in range(len(LTrain))])
            start = time.time()
            errorlst = []
            for p in range(len(XTrain)):
                x, y = XTrain[p], LTrain[p]
                if self.dodropout:
                    self.dropout()
                h0, a1, h1, a2, h2 = self.forwpass(x, self.W1, self.W2)
                errorlst.append(np.linalg.norm(y - h2))
                DeltaW1, DeltaW2 = self.backprop(y, h0, a1, h1, a2, h2, self.W2)
                self.W1 -= lrate_temp * DeltaW1
                self.W2 -= lrate_temp * DeltaW2
            if epochs % 10 == 0 and epochs > 0:
                lrate_temp *= 1. / (1. + (0.01 * epochs))
            self.progbarpurcentage += self.singlestep
            self.gui.progressBar.setValue(self.progbarpurcentage)
            tempstr = 'Training epoch n. {} - Loss : {} -- Time : {} s\n\n'
            toprint = (tempstr.format(epochs + 1, np.round(np.median(errorlst), 4),round(time.time() - start, 4)))
            self.textoutput += toprint
            self.logs += toprint
            self.gui.showterminal(self.textoutput)
            self.guiapp.processEvents()
            print(toprint)
        return self.W1, self.W2

    def predict(self, image, W1, W2):
        """
        Label prediction.
        :param image: Input image. (vector)
        :param W1: Weight of the first layer. (matrix)
        :param W2: Weight of the last layer. (matrix)
        :return: The label of the predicted input image.
        """
        return np.argmax(self.forwpass(image, W1, W2)[4])

    def crossValidation(self):
        """
        Cross-Validation
        :return: Numpy not normalized confusion matrix.
        """
        gscore = []
        for crossit in range(self.crossit):
            self.processrunning = True
            if self.useloaded:
                W1, W2 = self.loadedW1, self.loadedW2
            else:
                W1, W2 = self.trainperceptron()
            if crossit == self.crossit - 1 and self.shouldisave:
                self.sendweights = (W1, W2)
            np.random.shuffle(self.testset)
            XTest = self.testset[:, 1:]
            LTest = self.testset[:, 0]
            counter = 0
            for i in range(len(XTest)):
                prediction = self.predict(XTest[i], W1, W2)
                real = LTest[i]
                self.confusion[real][prediction] += 1
                if prediction == real:
                    counter += 1
            tempscore = (counter / len(XTest)) * 100
            gscore.append(tempscore)
            first = '\n[ TRAINING INFO ] - {} Training iterations completed\n'.format(crossit + 1)
            tempstr = '[ TEST INFO ] - Score on given test-set ({} items) : {} %\n\n\n'
            second = first + tempstr.format(len(XTest), tempscore)
            self.textoutput += second
            self.logs += second
            self.gui.showterminal(self.textoutput)
            self.guiapp.processEvents()
            print(first)
        endmessage = '\n[ GLOBAL SCORE ] Cross-Validation score on {} C.V. It. ( {} complete learning iterations ) : {} %\n\n'
        self.gui.progressBar.setValue(100)
        endmessage = endmessage.format(self.crossit, self.crossit * self.epochs, np.median(np.array(gscore)))
        print(endmessage)
        self.textoutput += endmessage
        self.logs += endmessage
        self.gui.showterminal(self.textoutput)
        self.processrunning = False
        self.guiapp.processEvents()
        return np.divide(self.confusion, self.crossit)
