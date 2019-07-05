import pandas as pd
import joblib
import cv2
from skimage import feature
import numpy as np
from scipy.spatial import distance

# Send image and pkl file to get prediction from image


class PredictFace:
    def __init__(self, image, datasetPath):
        self.image = image
        self.datasetPath = datasetPath

    def predict(self):
        flag = 0
        eps = 1e-7
        nds = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

        grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # lbph calculation
        lbp = feature.local_binary_pattern(grey, 8, 8, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        for i in range(len(hist)):
            hist[i] = hist[i] * 1000
        hist = hist.astype("int")

        # dataset creator
        nds.loc[flag] = [hist[0], hist[1],
                        hist[2], hist[3],
                         hist[4], hist[5],
                         hist[6], hist[7],
                         hist[8], hist[9]]
        # test data
        tds = pd.read_csv(self.datasetPath)
        res = ''
        b = nds.iloc[0]
        b = b.ravel()
        Dist = 10000
        flag = 0

        for i in range(len(tds)):
            a = tds.iloc[i]
            a = a[:-1]
            a = a.ravel()
            dst = distance.euclidean(a, b)
            if dst < Dist:
                flag = i
                Dist = dst
                if dst < 17:
                    res = tds.iloc[flag, -1]
                    print(dst)
                else:
                    res = "notFound"
                    print(dst)
        return res

if __name__ == '__main__':
    # create and show mainWindow
    path = "faceFeature.csv"
    img = cv2.imread("Suvam.jpg")
    mainWindow = PredictFace(img, path)
    result = mainWindow.predict()
    print(result)
