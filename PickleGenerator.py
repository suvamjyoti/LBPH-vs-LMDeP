
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from sklearn import svm

# add dataset to get pickle file


class PickleGenerator:
    def __init__(self, datasetpath):
        self.datasetpath = datasetpath

    def pklCreate(self):
        print("start1")
        nds = pd.read_csv(self.datasetpath)

        print("start2")
        train_X = nds[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]

        print("start3")
        train_y = nds.name

        print("start4")
        model = svm.SVC(kernel='linear', gamma='scale')
        print("start5")
        model.fit(train_X, train_y)
        print("start6")
        joblib.dump(model, 'FACE_LBPH_RF.pkl')
        print("finish")




