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

        y1.append(self.euclidianDistance(self.face94))
        y1.append(self.euclidianDistance(self.face95))
        y1.append(self.euclidianDistance(self.face97))
        y1.append(self.euclidianDistance(self.iitd))

        y2.append(self.supportVectormachine(self.face94))
        y2.append(self.supportVectormachine(self.face95))
        y2.append(self.supportVectormachine(self.face97))
        y2.append(self.supportVectormachine(self.iitd))

        y3.append(self.kNearestNeighbour(self.face94))
        y3.append(self.kNearestNeighbour(self.face95))
        y3.append(self.kNearestNeighbour(self.face97))
        y3.append(self.kNearestNeighbour(self.iitd))

        y4.append(self.randomForest(self.face94))
        y4.append(self.randomForest(self.face95))
        y4.append(self.randomForest(self.face97))
        y4.append(self.randomForest(self.iitd))

        plt.plot(x, y1, 'r', label='EucDist')
        plt.plot(x, y2, 'g', label='SVM')
        plt.plot(x, y3, 'b', label='KNN')
        plt.plot(x, y4, 'k', label='RandF')
        plt.legend()
        plt.ylabel('ACCURACY')
        plt.xlabel('FACE_DATASET')
        plt.title('LBPH')
        plt.savefig('a.png')
        plt.close()
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
        print('e')

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
        print('f')

        train, test = train_test_split(data, test_size=0.3)

        train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        train_y = train.name
        test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]  # taking test data features
        test_y = test.name

        model = svm.SVC(kernel='linear', gamma='scale')
        model.fit(train_X, train_y)

        y_pred = model.predict(test_X)
        #result = precision_recall_fscore_support(test_y, y_pred, average='macro')
        accu = metrics.accuracy_score(test_y, y_pred)
        print(accu)
        return accu

    def randomForest(self, data):
        print('g')

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
        print('su')

    def DrawComparision(self):

        if self.classifier == 1:
            a1, a2, a3 = self.randomForest(self.data1, 1)
            b1, b2, b3 = self.randomForest(self.data2, 2)
            plt.title('Random Forest')

        elif self.classifier == 2:
            a1, a2, a3 = self.euclidianDistance(self.data1)
            b1, b2, b3 = self.euclidianDistance(self.data2)
            plt.title('Euclidian Distance')

        elif self.classifier == 3:
            a1, a2, a3 = self.supportVectormachine(self.data1, 1)
            b1, b2, b3 = self.supportVectormachine(self.data2, 2)
            plt.title('Suport Vector Machine')

        else:
            a1, a2, a3 = self.kNearestNeighbour(self.data1, 1)
            b1, b2, b3 = self.kNearestNeighbour(self.data2, 2)
            plt.title('K Nearest Neighbour')
        # data to plot
        n_groups = 3
        means_frank = (a1, a2, a3)
        means_guido = (b1, b2, b3)

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

        plt.xlabel('parameter')
        plt.ylabel('Scores')
        plt.xticks(index + bar_width, ('precision', 'recall', 'accuracy'))
        plt.legend()
        plt.tight_layout()
        plt.savefig('a.png')
        img = cv2.imread('a.png')
        small = cv2.resize(img, (800, 400))

        return small
       
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
            test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
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
            test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
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
            print('rf4')
            train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                             '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]
            print('rf5')
            test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                           '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]

        train_y = train.name
        test_y = test.name
        model = RandomForestClassifier(n_estimators=30, bootstrap=True, max_features='sqrt')
        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        result = precision_recall_fscore_support(test_y, y_pred, average='macro')
        accu = metrics.accuracy_score(test_y, y_pred)
        print("result = " , result)
        return result[0], result[1], accu










