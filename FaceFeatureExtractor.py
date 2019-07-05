from skimage import feature
import numpy as np
import cv2
import pandas as pd
import os
import csv

row = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'name']


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def FeatureExtract(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        for i in range(len(hist)):
            hist[i] = hist[i] * 1000

        hist = hist.astype("int")
        # return the histogram of Local Binary Patterns
        return hist

class CreateDataset:
    def __init__(self, data, name):
        self.data = data
        self.name = name

    def datasetCreator(self):

        row = [self.data[0], self.data[1],
                        self.data[2], self.data[3],
                        self.data[4], self.data[5],
                        self.data[6], self.data[7],
                        self.data[8], self.data[9], self.name]
        return row


class FindFace():
    def __init__(self, img, name):
        self.img = img
        self.name = name
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T1 = LocalBinaryPatterns(8, 8)
        k = T1.FeatureExtract(grey)
        if not os.path.exists('faceFeature.csv'):
            open("faceFeature.csv", "w")
            with open('faceFeature.csv','a')as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)

        d2 = CreateDataset(k, self.name)
        row1= d2.datasetCreator()
        with open('faceFeature.csv','a')as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row1)
        csvFile.close()



