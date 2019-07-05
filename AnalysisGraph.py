import matplotlib.pyplot as plt
import cv2
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


class DataAnalysis:

    def __init__(self, face94, face95, face97, iitd):
        self.face94 = face94
        self.face95 = face95
        self.face97 = face97
        self.iitd = iitd

    def drawGraph(self):

        x = ['Face94', 'Face95', 'Face97', 'IITD']
        y1 = []
        y2 = []
        y3 = []
        y4 = []

        y1[0] = self.euclidianDistance(self.face94)
        y1[1] = self.euclidianDistance(self.face95)
        y1[2] = self.euclidianDistance(self.face97)
        y1[3] = self.euclidianDistance(self.iitd)

        y2[0] = self.supportVectormachine(self.face94)
        y2[1] = self.supportVectormachine(self.face95)
        y2[2] = self.supportVectormachine(self.face97)
        y2[3] = self.supportVectormachine(self.iitd)

        y3[0] = self.kNearestNeighbour(self.face94)
        y3[1] = self.kNearestNeighbour(self.face95)
        y3[2] = self.kNearestNeighbour(self.face97)
        y3[3] = self.kNearestNeighbour(self.iitd)

        y4[0] = self.randomForest(self.face94)
        y4[1] = self.randomForest(self.face95)
        y4[2] = self.randomForest(self.face97)
        y4[3] = self.randomForest(self.iitd)

        plt.plot(x, y1, 'r', label='EucDist')
        plt.plot(x, y2, 'g', label='SVM')
        plt.plot(x, y3, 'b', label='KNN')
        plt.plot(x, y4, 'k', label='RandF')
        plt.legend()
        plt.ylabel('ACCURACY')
        plt.xlabel('FACE_DATASET')
        plt.title('LBPH')
        plt.savefig('a.png')
        img = cv2.imread('a.png')

        return img

    def euclidianDistance(self, data):
        train, test = train_test_split(data, test_size=0.3)

        # program starts
        test_y = []
        y_pred = []
        res = ''
        for i in range(test.shape[0]):
                Dist = 10000
                a = test.iloc[i]
                a = a[:-1]
                for m in range(train.shape[0]):
                    b = train.iloc[m]
                    b = b[:-1]
                    dst = distance.euclidean(a, b)
                    if dst < Dist:
                        Dist = dst
                        res = train.iloc[m, -1]

                y_pred.append(res)
                res2 = test.iloc[i, -1]
                test_y.append(res2)
        result = precision_recall_fscore_support(test_y, y_pred, average='macro')
        return result[2]

    def kNearestNeighbour(self, data):

        train, test = train_test_split(data, test_size=0.3)

        train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        train_y = train.name
        test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        test_y = test.name

        scaler = StandardScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(train_X, train_y)
        y_pred = classifier.predict(test_X)
        result = precision_recall_fscore_support(test_y, y_pred, average='macro')
        return result[2]

    def supportVectormachine(self, data):

        train, test = train_test_split(data, test_size=0.3)

        train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        train_y = train.name
        test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]  # taking test data features
        test_y = test.name

        model = svm.SVC(kernel='linear', gamma='scale')
        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        result = precision_recall_fscore_support(test_y, y_pred, average='macro')
        return result[2]

    def randomForest(self, data):

        train, test = train_test_split(data, test_size=0.3)

        train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        train_y = train.name
        test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]  # taking test data features
        test_y = test.name
        model = RandomForestClassifier(n_estimators=30, bootstrap=True, max_features='sqrt')
        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        result = precision_recall_fscore_support(test_y, y_pred, average='macro')

        return result[2]


class ComparisionGraph:

    def __init__(self, data1, data2, size, classifier):
        self.data1 = data1
        self.data2 = data2
        self.size = size
        self.classifier = classifier


    def DrawComparision(self):

        if self.classifier == 1:
            start = time.process_time()
            a1 = self.randomForest(self.data1, 1)
            e1 = time.process_time() - start
            a1.append(e1)
            a2 = self.randomForest(self.data2, 2)
            e2 = time.process_time() - e1
            a2.append(e2)

        elif self.classifier == 2:
            start = time.process_time()
            a1 = self.euclidianDistance(self.data1)
            e1 = time.process_time() - start
            a1.append(e1)
            a2 = self.euclidianDistance(self.data2)
            e2 = time.process_time() - e1
            a2.append(e2)


        elif self.classifier == 3:
            start = time.process_time()
            a1 = self.supportVectormachine(self.data1, 1)
            e1 = time.process_time() - start
            a1.append(e1)
            a2 = self.supportVectormachine(self.data2, 2)
            e2 = time.process_time() - e1
            a2.append(e2)

        else:
            start = time.process_time()
            a1 = self.kNearestNeighbour(self.data1, 1)
            e1 = time.process_time() - start
            a1.append(e1)
            a2 = self.kNearestNeighbour(self.data2, 2)
            e2 = time.process_time() - e1
            a2.append(e2)

        # data to plot
        n_groups = 4
        means_frank = a1
        means_guido = a2

        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, means_frank, bar_width,
                         alpha=opacity,
                         color='b',
                         label='LBPH')

        rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                         alpha=opacity,
                         color='g',
                         label='LMDEP')

        plt.xlabel('Person')
        plt.ylabel('Scores')
        plt.title('Scores by person')
        plt.xticks(index + bar_width, ('precision', 'recall', 'accuracy', 'time'))
        plt.legend()
        plt.tight_layout()
        plt.savefig('a.png')
        img = cv2.imread('a.png')

        return img
       
    def euclidianDistance(self, data):
        tsize = 1 - (self.size/100)
        train, test = train_test_split(data, test_size=tsize)

        # program starts
        test_y = []
        y_pred = []
        res = ''
        for i in range(test.shape[0]):
                Dist = 10000
                a = test.iloc[i]
                a = a[:-1]
                for m in range(train.shape[0]):
                    b = train.iloc[m]
                    b = b[:-1]
                    dst = distance.euclidean(a, b)
                    if dst < Dist:
                        Dist = dst
                        res = train.iloc[m, -1]

                y_pred.append(res)
                res2 = test.iloc[i, -1]
                test_y.append(res2)
        result = precision_recall_fscore_support(test_y, y_pred, average='macro')
        accu = metrics.accuracy_score(test_y, y_pred)
        return result[0], result[1], accu

    def kNearestNeighbour(self, data, type):

        tsize = 1 - (self.size/100)
        train, test = train_test_split(data, test_size=tsize)

        if type == 1:
            train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
            test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        else:
            train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                             '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]
            test_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                            '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]

        train_y = train.name
        test_y = test.name

        scaler = StandardScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(train_X, train_y)
        y_pred = classifier.predict(test_X)
        result = precision_recall_fscore_support(test_y, y_pred, average='macro')
        accu = metrics.accuracy_score(test_y, y_pred)
        return result[0], result[1], accu

    def supportVectormachine(self, data, type):

        tsize = 1 - (self.size/100)
        train, test = train_test_split(data, test_size=tsize)

        if type == 1:
            train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
            test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        else:
            train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                             '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]
            test_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                            '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]

        train_y = train.name
        test_y = test.name

        model = svm.SVC(kernel='linear', gamma='scale')
        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        result = precision_recall_fscore_support(test_y, y_pred, average='macro')
        accu = metrics.accuracy_score(test_y, y_pred)
        return result[0], result[1], accu

    def randomForest(self, data, type):

        tsize = 1 - (self.size/100)
        train, test = train_test_split(data, test_size=tsize)

        if type == 1:
            train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
            test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        else:
            train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                             '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]
            test_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                            '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]

        train_y = train.name
        test_y = test.name
        model = RandomForestClassifier(n_estimators=30, bootstrap=True, max_features='sqrt')
        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        result = precision_recall_fscore_support(test_y, y_pred, average='macro')
        accu = metrics.accuracy_score(test_y, y_pred)
        return result[0], result[1], accu










