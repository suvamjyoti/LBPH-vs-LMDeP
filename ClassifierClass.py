from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

#-----------------------------------------------------------------------------------------------------------------------

## send data after reading it using pandas from csv file
## for eg - nds = pd.read_csv("E:/ImageProcessing/Mangalore/DataBase/train.csv")
## then send nds

#-----------------------------------------------------------------------------------------------------------------------


class Classifier:

    def __init__(self, data, type, size):
        self.data = data
        self.type = type
        self.size = size
        print(self.size)

    def euclidianDistance(self):
        tsize = 1 - (self.size/100)
        train, test = train_test_split(self.data, test_size=tsize)

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

        result = precision_recall_fscore_support(test_y,y_pred,average="macro")
        accu = metrics.accuracy_score(test_y, y_pred)
        return result[0], result[1], accu

    def kNearestNeighbour(self):

        tsize = 1 - (self.size / 100)
        train, test = train_test_split(self.data, test_size=tsize)

        if self.type == 1:
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

        result = precision_recall_fscore_support(test_y, y_pred,average="macro")
        accu = metrics.accuracy_score(test_y, y_pred)
        return result[0], result[1], accu

    def supportVectormachine(self):

        tsize = 1 - (self.size / 100)
        train, test = train_test_split(self.data, test_size=tsize)

        if self.type == 1:
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

        result = precision_recall_fscore_support(test_y, y_pred,average="macro")
        accu = metrics.accuracy_score(test_y, y_pred)
        return result[0], result[1], accu

    def randomForest(self):

        tsize = 1 - (self.size / 100)
        train, test = train_test_split(self.data, test_size=tsize)

        if self.type == 1:
            train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
            test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        else:
            train_X = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                             '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]
            test_X = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                           '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]


        train_y = train.name
        test_y = test.name

        model = RandomForestClassifier(n_estimators=30, bootstrap=True, max_features='sqrt')
        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)

        result = precision_recall_fscore_support(test_y, y_pred, average="macro")
        accu = metrics.accuracy_score(test_y, y_pred)
        print("accu = ", accu)
        return result[0], result[1], accu

if __name__ == '__main__':
    # create and show mainWindow
    nds = pd.read_csv("FaceFeatures.csv")
    mainWindow = Classifier(nds,1)
    result =mainWindow.randomForest()
    print(result)






