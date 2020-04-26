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


def error(errorPrediction, total):
    print("Error is", errorPrediction, "out of Total of ", total)
    errorRate = (errorPrediction * 100) / total
    print(errorRate, "%")


if __name__ == '__main__':
    print("TRAINING OUR MODEL FIRST")
    PERCENT_INCREMENT = 10
    POSSIBLE_VALUES = [0, 1]  # BINARY

    # Reading the files
    samples = Samples()
    samples.readFiles()

    dataClassifier = DataClassifier()

    # Extracting Features from the Training Data
    dataset = 0
    featureValueListForAllTrainingImages, actualLabelForTrainingList = \
        dataClassifier.extractFeatures(samples.train_lines_itr, samples.train_labelsLines_itr)

    TOTALDATASET = len(actualLabelForTrainingList)
    INCREMENTS = int(TOTALDATASET * PERCENT_INCREMENT / 100)
    PERCEPTRON_TIME = {}

    # Initialization of Classifiers
    perceptronClassifier = PerceptronClassifier(dataClassifier.FEATURES, dataClassifier.LABELS)
    naiveBayesClassifier = NaiveBayesClassifier(dataClassifier.FEATURES, dataClassifier.LABELS, POSSIBLE_VALUES)

    while dataset < TOTALDATASET:

        startTimer = time.time()

        featureValueList_currentTrainingImages = featureValueListForAllTrainingImages[dataset:dataset+INCREMENTS]
        actualLabel_currentTrainingImages = actualLabelForTrainingList[dataset:dataset+INCREMENTS]

        print("Training ON {0} to {1} data".format(dataset, dataset+INCREMENTS))
        ImageFeatureLabelZipList = zip(featureValueList_currentTrainingImages, actualLabel_currentTrainingImages)

        ''' ####################  TRAINING PHASE FOR PERCEPTRON ############# '''
        for featureValueListPerImage, actualLabel in ImageFeatureLabelZipList:
            perceptronClassifier.runModel(True, featureValueListPerImage, actualLabel)

        ''' ####################  TRAINING PHASE FOR NAIVE BYES ############# '''
        naiveBayesClassifier.constructLabelsProbability(actualLabel_currentTrainingImages)
        naiveBayesClassifier.constructFeaturesProbability(featureValueList_currentTrainingImages,
                                                          actualLabel_currentTrainingImages,
                                                          POSSIBLE_VALUES)

        endTimer = time.time()

        ''' ####################  TESTING PHASE ############# '''
        samples.initTestIters()

        print("TESTING our model that is TRAINED ON {0} to {1} data".format(0, dataset+INCREMENTS))

        perceptron_errorPrediction = naiveByes_errorPrediction = total = 0
        featureValueListForAllTestingImages, actualLabelList = \
            dataClassifier.extractFeatures(samples.test_lines_itr, samples.test_labelsLines_itr)

        for featureValueListPerImage, actualLabel in zip(featureValueListForAllTestingImages, actualLabelList):
            perceptron_errorPrediction += perceptronClassifier.runModel(False, featureValueListPerImage, actualLabel)
            naiveByes_errorPrediction += naiveBayesClassifier.testModel(featureValueListPerImage, actualLabel)
            total += 1

        error(perceptron_errorPrediction, total)
        error(naiveByes_errorPrediction, total)

        dataset += INCREMENTS

    samples.closeFiles()

def dummyplot():
    plt.plot([1, 2, 3], [2, 3, 4])
    plt.ylabel('Error Rate')
    plt.xlabel('DataSet')
    plt.show()



