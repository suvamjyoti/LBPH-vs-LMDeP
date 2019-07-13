import numpy as np
import math
import cv2
import pandas as pd

class LMDeP():

    def __init__(self, imSource, imName, max):
        # store the number of points and radius
        self.imsource = imSource
        self.imname = imName
        self.max = max

    def featureExtractLmdep(self):

        flag = 0
        nds = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                    '11', '12', '13', '14', '15', '16', '17', '18', '19', 'name'])

        if len(self.imsource)<self.max:
            self.max = len(self.imsource)
        for i in range(self.max):
            print(i)
            image = cv2.imread(self.imsource[i])
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            a = np.asarray(grey, dtype='int32')

            angFeature = self.AngularMeanExcitation(a)
            print(angFeature[0])
            radFeature = self.RadialMeanExcitation(a)
            print("rad featur", radFeature)
            nds.loc[flag] = [angFeature[0], angFeature[1], angFeature[2], angFeature[3],
                             angFeature[4], angFeature[5], angFeature[6], angFeature[7],
                             angFeature[8], angFeature[9], radFeature[0], radFeature[1],
                             radFeature[2], radFeature[3], radFeature[4], radFeature[5],
                             radFeature[6], radFeature[7], radFeature[8], radFeature[9], self.imname[i]]
            print("here")
            flag += 1

        path = 'FaceFeatures.csv'
        nds.to_csv(path)


        return path

    def AngularMeanExcitation(self, img):
        print("hey")
        w, h = img.shape[:2]

        res = np.zeros((w, h), np.uint32)
        res2 = np.zeros((w, h), np.uint32)

        for i in range(1, w-1):
            for j in range(1, h-1):

                mvalue = img[i, j]
                n = (img[i - 1, j - 1] + img[i, j - 1] + img[i + 1, j - 1]) / 3
                s = (img[i - 1, j + 1] + img[i, j + 1] + img[i + 1, j + 1]) / 3
                e = (img[i + 1, j - 1] + img[i + 1, j] + img[i + 1, j + 1]) / 3
                w = (img[i - 1, j - 1] + img[i - 1, j] + img[i - 1, j + 1]) / 3
                se = (img[i + 1, j] + img[i + 1, j + 1] + img[i, j + 1]) / 3
                sw = (img[i - 1, j] + img[i - 1, j + 1] + img[i, j + 1]) / 3
                ne = (img[i, j - 1] + img[i + 1, j - 1] + img[i + 1, j]) / 3
                nw = (img[i, j - 1] + img[i - 1, j - 1] + img[i, j - 1]) / 3

                res[i, j-1] = math.atan(mvalue - n)
                res[i, j+1] = math.atan(mvalue - s)
                res[i+1, j] = math.atan(mvalue - e)
                res[i-1, j] = math.atan(mvalue - w)
                res[i + 1, j + 1] = math.atan(mvalue - se)
                res[i - 1, j + 1] = math.atan(mvalue - sw)
                res[i + 1, j - 1] = math.atan(mvalue - ne)
                res[i - 1, j - 1] = math.atan(mvalue - nw)

                if res[i, j-1] > 0:
                    res[i, j - 1] = 1
                else:
                    res[i, j - 1] = 0

                if res[i, j+1] > 0:
                    res[i, j + 1] = 1
                else:
                    res[i, j + 1] = 0

                if res[i+1, j] > 0:
                    res[i+1, j] = 1
                else:
                    res[i + 1, j] = 0

                if res[i-1, j] > 0:
                    res[i-1, j] = 1
                else:
                    res[i-1, j] = 0

                if res[i+1, j+1] > 0:
                    res[i+1, j + 1] = 1
                else:
                    res[i+1, j + 1] = 0

                if res[i-1, j+1] > 0:
                    res[i-1, j+1] = 1
                else:
                    res[i-1, j+1] = 0

                if res[i+1, j-1] > 0:
                    res[i+1, j - 1] = 1
                else:
                    res[i+1, j - 1] = 0

                if res[i-1, j-1] > 0:
                    res[i-1, j - 1] = 1
                else:
                    res[i-1, j - 1] = 0

                arr = [res[i, j - 1], res[i, j + 1], res[i + 1, j], res[i - 1, j],
                       res[i + 1, j + 1], res[i - 1, j + 1],
                       res[i + 1, j - 1], res[i - 1, j - 1]]
                Sum1 = 0
                for m in range(8):
                    k = 2**m
                    Sum1 += arr[m]*k

                res2[i, j] = Sum1

        (hist, _) = np.histogram(res2.ravel(), bins=10)

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        for i in range(len(hist)):
            hist[i] = hist[i] * 1000

        hist = hist.astype("int")
        # return the histogram of Local Binary Patterns

        return hist

    def RadialMeanExcitation(self, img):
        print("bye")
        w, h = img.shape[:2]

        res = np.zeros((w, h), np.uint32)
        res2 = np.zeros((w, h), np.uint32)

        for i in range(3, w-3):
            for j in range(3, h-3):

                mvalue = img[i, j]
                n = (img[i, j - 1] + img[i, j - 2] + img[i, j - 3]) / 3
                s = (img[i, j + 1] + img[i, j + 2] + img[i, j + 3]) / 3
                e = (img[i + 1, j] + img[i + 2, j] + img[i + 3, j]) / 3
                w = (img[i - 1, j] + img[i - 2, j] + img[i - 3, j]) / 3
                se = (img[i + 1, j + 1] + img[i + 2, j + 2] + img[i + 3, j + 3]) / 3
                sw = (img[i - 1, j + 1] + img[i - 2, j + 2] + img[i - 3, j + 3]) / 3
                ne = (img[i + 1, j - 1] + img[i + 2, j - 2] + img[i + 3, j - 3]) / 3
                nw = (img[i - 1, j - 1] + img[i - 2, j - 2] + img[i - 3, j - 3]) / 3

                res[i, j-1] = math.atan(mvalue - n)
                res[i, j+1] = math.atan(mvalue - s)
                res[i+1, j] = math.atan(mvalue - e)
                res[i-1, j] = math.atan(mvalue - w)
                res[i + 1, j + 1] = math.atan(mvalue - se)
                res[i - 1, j + 1] = math.atan(mvalue - sw)
                res[i + 1, j - 1] = math.atan(mvalue - ne)
                res[i - 1, j - 1] = math.atan(mvalue - nw)

                if res[i, j-1] > 0:
                    res[i, j - 1] = 1
                else:
                    res[i, j - 1] = 0

                if res[i, j+1] > 0:
                    res[i, j + 1] = 1
                else:
                    res[i, j + 1] = 0

                if res[i+1, j] > 0:
                    res[i+1, j] = 1
                else:
                    res[i + 1, j] = 0

                if res[i-1, j] > 0:
                    res[i-1, j] = 1
                else:
                    res[i-1, j] = 0

                if res[i+1, j+1] > 0:
                    res[i+1, j + 1] = 1
                else:
                    res[i+1, j + 1] = 0

                if res[i-1, j+1] > 0:
                    res[i-1, j+1] = 1
                else:
                    res[i-1, j+1] = 0

                if res[i+1, j-1] > 0:
                    res[i+1, j - 1] = 1
                else:
                    res[i+1, j - 1] = 0

                if res[i-1, j-1] > 0:
                    res[i-1, j - 1] = 1
                else:
                    res[i-1, j - 1] = 0

                arr = [res[i, j - 1], res[i, j + 1], res[i + 1, j], res[i - 1, j],
                       res[i + 1, j + 1], res[i - 1, j + 1],
                       res[i + 1, j - 1], res[i - 1, j - 1]]
                Sum1 = 0
                for m in range(8):
                    k = 2**m
                    Sum1 += arr[m]*k

                res2[i, j] = Sum1

        (hist, _) = np.histogram(res2.ravel(), bins=10)

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        for i in range(len(hist)):
            hist[i] = hist[i] * 1000

        hist = hist.astype("int")
        # return the histogram of Local Binary Patterns

        return hist

