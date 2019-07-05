from skimage import feature
import numpy as np
import cv2
import pandas as pd

class LocalBinaryPatterns:
    def __init__(self, imSource, imName, max):
        # store the number of points and radius
        self.numPoints = 8
        self.radius = 8
        self.imsource = imSource
        self.imname = imName
        self.max = max


    def featureExtractLbph(self, eps=1e-7):

        flag = 0
        nds = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'name'])
        print(self.imsource[4])
        #face_cascade = cv2.CascadeClassifier('HAARFace.xml')

        if len(self.imsource)<self.max:
            self.max = len(self.imsource)
        for i in range(self.max):
            print(i)
            img = cv2.imread(self.imsource[i])

            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #faces = face_cascade.detectMultiScale(grey, 1.3, 5)

            #for (x, y, w, h) in faces:
                #img = grey[y:y + h, x:x + w]

            lbp = feature.local_binary_pattern(grey, self.numPoints,
                                               self.radius, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, self.numPoints + 3),
                                     range=(0, self.numPoints + 2))

            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)

            for k in range(len(hist)):
                hist[k] = hist[k] * 1000

            hist = hist.astype("int")
            # return the histogram of Local Binary Patterns

            nds.loc[flag] = [hist[0], hist[1],
                            hist[2], hist[3],
                            hist[4], hist[5],
                            hist[6], hist[7],
                            hist[8], hist[9], self.imname[i]]
            flag += 1

        path = 'FaceFeatures.csv'
        nds.to_csv(path)

        return path
