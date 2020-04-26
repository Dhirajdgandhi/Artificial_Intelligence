import math
import time
import matplotlib.pyplot as plt
from naivebyes import NaiveBayesClassifier
from perceptron import PerceptronClassifier

from samples import Samples


class DataClassifier:
    def __init__(self, imgHeight=20, imgWidth=29, LABELS=10, pixelChars=None):
        if pixelChars is None:
            pixelChars = ['#', '+']
        self.pixelGrid = 1
        self.imgHeight = imgHeight
        self.FEATURES = math.ceil((imgHeight * imgWidth) / self.pixelGrid)
        self.LABELS = LABELS
        self.pixelChars = pixelChars
        self.FileObject = None
        self.LabelFileObject = None

    def countPixels(self, line):
        count = 0
        if not isinstance(line, list):
            line = list(line)

        for char in line:
            if char in self.pixelChars:
                count += 1

        return count

    def extractFeaturesPerLine(self, line, row):
        gridList = []
        featureValueList = []

        for startIndexOfGrid in range(0, len(line), self.pixelGrid):
            gridList.append(line[startIndexOfGrid:startIndexOfGrid + self.pixelGrid])

        # col = 0
        for grid in gridList:
            # Count the number of chars in this grid and add the count to respective index of FEATURE
            # indexOfFeature = row + col
            featureValueList.append(self.countPixels(grid))

        return featureValueList

    def extractFeatures(self, lines_itr, labelsLines_itr):
        imageLine = lines_itr.__next__()

        totalImages = 0
        featureValueListPerImage = [1]
        featureValueListForAllTestingImages = []
        actualLabelList = []

        try:
            while imageLine:
                # Skipping the blank lines
                while imageLine and self.countPixels(imageLine) == 0:
                    imageLine = lines_itr.__next__()

                # Scanning image pixels
                for i in range(0, self.imgHeight):
                    featureValueListPerImage.extend(self.extractFeaturesPerLine(imageLine, i))
                    # print(featureValueList)
                    imageLine = lines_itr.__next__()

                totalImages += 1
                actualLabel = labelsLines_itr.__next__()

                featureValueListForAllTestingImages.append(featureValueListPerImage)
                actualLabelList.append(int(actualLabel))

                # Re-init the feature score
                featureValueListPerImage = [1]
        except StopIteration:
            # print("End of File")
            pass

        return featureValueListForAllTestingImages, actualLabelList


if __name__ == '__main__':
    print("TRAINING OUR MODEL FIRST")
    PERCENT_INCREMENT = 10
    POSSIBLE_VALUES = 2

    samples = Samples()
    dataClassifier = DataClassifier()
    perceptronClassifier = PerceptronClassifier(dataClassifier.FEATURES, dataClassifier.LABELS)

    samples.readFiles()
    dataset = 0
    featureValueListForAllTrainingImages, actualLabelForTrainingList = \
        dataClassifier.extractFeatures(samples.train_lines_itr, samples.train_labelsLines_itr)

    # Naives Byes Algorithm
    naiveBayesClassifier = NaiveBayesClassifier()
    naiveBayesClassifier.constructLabelsProbability(actualLabelForTrainingList)
    POSSIBLE_VALUES = [0, 1]  # BINARY
    naiveBayesClassifier.constructFeaturesProbability(featureValueListForAllTrainingImages, actualLabelForTrainingList,
                                                      POSSIBLE_VALUES)

    TOTALDATASET = len(featureValueListForAllTrainingImages)
    INCREMENTS = int(TOTALDATASET * PERCENT_INCREMENT / 100)
    PERCEPTRON_TIME = {}

    while dataset < TOTALDATASET:

        startTimer = time.time()

        print("Training ON {0} to {1} data".format(dataset, dataset+INCREMENTS))
        ImageLabelZipList = zip(featureValueListForAllTrainingImages[dataset:dataset+INCREMENTS], actualLabelForTrainingList[dataset:dataset+INCREMENTS])

        for featureValueListPerImage, actualLabel in ImageLabelZipList:
            perceptronClassifier.runModel(True, featureValueListPerImage, actualLabel)

        endTimer = time.time()

        print("TESTING our model that is TRAINED ON {0} to {1} data".format(0, dataset+INCREMENTS))

        errorPrediction = 0
        errorPrediction1 = 0
        total = 0
        featureValueListForAllTestingImages, actualLabelList = dataClassifier.extractFeatures(samples.test_lines_itr, samples.test_labelsLines_itr)

        for featureValueListPerImage, actualLabel in zip(featureValueListForAllTestingImages, actualLabelList):
            errorPrediction += perceptronClassifier.runModel(False, featureValueListPerImage, actualLabel)
            # errorPrediction1 += naiveBayesClassifier.testModel(featureValueListPerImage, actualLabel)
            total += 1

        samples.initTestIters()

        print("Error is", errorPrediction, "out of Total of ", total)
        errorRate = (errorPrediction * 100) / total
        print(errorRate, "%")

        print("Error is", errorPrediction1, "out of Total of ", total)
        errorRate1 = (errorPrediction1 * 100) / total
        print(errorRate1, "%")

        dataset += INCREMENTS

        PERCEPTRON_TIME[dataset] = ((endTimer-startTimer), errorRate)

    samples.closeFiles()

def dummyplot():
    plt.plot([1, 2, 3], [2, 3, 4])
    plt.ylabel('Error Rate')
    plt.xlabel('DataSet')
    plt.show()



