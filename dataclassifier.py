import math
import sys
import time
import argparse
import statistics
from knn import KNN
import matplotlib.pyplot as plt
from naivebyes import NaiveBayesClassifier
from perceptron import PerceptronClassifier
from error_plot import Error
from samples import Samples


def mean_standard_deviation(errorRate, name):
    if len(errorRate) > 1:
        mean = statistics.mean(errorRate)
        standard_deviation = statistics.stdev(errorRate)
        print(name, " mean = ", mean, " and Standard Deviation = ", standard_deviation)
        return mean


class DataClassifier:
    def __init__(self, imgHeight, imgWidth, LABELS, pixelChars, pixelGrid):
        if pixelChars is None:
            pixelChars = ['#', '+']
        self.pixelGrid = pixelGrid
        self.imgHeight = imgHeight
        self.imgWidth = imgWidth
        self.FEATURES = math.ceil((imgHeight - self.pixelGrid + 1) * (imgWidth - self.pixelGrid + 1))
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

        for startIndexOfGrid in range(0, len(line), 1):
            gridList.append(line[startIndexOfGrid:startIndexOfGrid + 1])

        # col = 0
        for grid in gridList:
            # Count the number of chars in this grid and add the count to respective index of FEATURE
            # indexOfFeature = row + col
            featureValueList.append(self.countPixels(grid))

        return featureValueList

    def splitImageLineFeaturesIntoGridFeatures(self, imageLinesList, gridSize):
        height_rows = self.imgHeight + 1 - gridSize
        width_rows = self.imgWidth + 1 - gridSize
        height_new_list = []

        for rowIndex in range(0, self.imgHeight):
            line = imageLinesList[rowIndex]
            width_new_list = []
            for gridStartIndex in range(0, width_rows):
                width_new_list.append(sum(line[gridStartIndex: gridStartIndex + gridSize]))
            height_new_list.append(width_new_list)

        featureListForImage = []
        for rowIndex in range(0, height_rows):
            for column in range(0, width_rows):
                sum1 = 0
                for rows in range(0, gridSize):
                    sum1 += height_new_list[rowIndex + rows][column]
                featureListForImage.append(sum1)

        return featureListForImage

    def extractFeatures(self, lines_itr, labelsLines_itr):
        imageLine = lines_itr.__next__()

        totalImages = 0
        featureValueListForAllTestingImages = []
        actualLabelList = []

        try:
            while imageLine:
                # Skipping the blank lines
                while imageLine and self.countPixels(imageLine) == 0:
                    imageLine = lines_itr.__next__()

                imageLinesList = []
                # Scanning image pixels
                for i in range(0, self.imgHeight):
                    imageLinesList.append(self.extractFeaturesPerLine(imageLine, i))
                    # print(featureValueList)
                    imageLine = lines_itr.__next__()

                featureValueListPerImage = self.splitImageLineFeaturesIntoGridFeatures(imageLinesList, gridSize)

                totalImages += 1
                actualLabel = labelsLines_itr.__next__()

                featureValueListForAllTestingImages.append(featureValueListPerImage)
                actualLabelList.append(int(actualLabel))

        except StopIteration:
            # print("End of File")
            pass

        return featureValueListForAllTestingImages, actualLabelList


def error(errorPrediction, total):
    errorRate = (errorPrediction * 100) / total
    print("Error is", errorPrediction, "out of Total of ", total, " : ", errorRate)
    return errorRate


FACE = "FACE"
DIGIT = "DIGIT"
DIR = "DIR"
HEIGHT = "HEIGHT"
WIDTH = "WIDTH"
LABEL = "LABEL"
PIXELS = "PIXELS"

if __name__ == '__main__':
    print("TRAINING OUR MODEL FIRST")
    # PERCENT_INCREMENT = 10

    perceptron_y = []
    bayes_y = []
    knn_y = []
    dataSetIncrements = []
    perceptron_time = []
    bayes_time = []
    knn_time = []
    perceptron_msd=[]
    bayes_msd=[]
    knn_msd=[]

    # inp = input("Type FACE or DIGIT")
    # gridSize = int(input("Value of Grid"))
    # inp = sys.argv[1]
    # gridSize = int(sys.argv[2])
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--gridSize')
    #Have to add from this
    parser.add_argument('--smoothingValue')
    parser.add_argument('--classifier')
    parser.add_argument('--percentIncrement')
    args = parser.parse_args()

    inp = args.input
    gridSize = int(args.gridSize)
    k_value = float(args.smoothingValue)
    PERCENT_INCREMENT = int(args.percentIncrement)
    POSSIBLE_VALUES = [x for x in range(0, gridSize * gridSize + 1)]

    map = {
        FACE: {
            DIR: 'data/facedata', HEIGHT: 68, WIDTH: 61, LABEL: 2, PIXELS: None
        },
        DIGIT: {
            DIR: 'data/digitdata', HEIGHT: 20, WIDTH: 29, LABEL: 10, PIXELS: None
        }
    }

    dataType = map.get(inp)
    samples = Samples(dataType.get(DIR))

    dataClassifier = DataClassifier(dataType.get(HEIGHT), dataType.get(WIDTH), dataType.get(LABEL),
                                    dataType.get(PIXELS), gridSize)
    perceptronClassifier = PerceptronClassifier(dataClassifier.FEATURES, dataClassifier.LABELS)

    samples.readFiles()

    # Extracting Features from the Training Data
    dataset = 0
    featureValueListForAllTrainingImages, actualLabelForTrainingList = \
        dataClassifier.extractFeatures(samples.train_lines_itr, samples.train_labelsLines_itr)

    TOTALDATASET = len(actualLabelForTrainingList)
    INCREMENTS = int(TOTALDATASET * PERCENT_INCREMENT / 100)
    PERCEPTRON_TIME = {}

    # Initialization of Classifiers
    perceptronClassifier = PerceptronClassifier(dataClassifier.FEATURES, dataClassifier.LABELS)
    naiveBayesClassifier = NaiveBayesClassifier(dataClassifier.FEATURES, dataClassifier.LABELS, POSSIBLE_VALUES, k_value)
    KNNClassifier = KNN(num_neighbors=20)

    featureValueListForAllTestingImages = actualTestingLabelList = []
    while dataset < TOTALDATASET:

        featureValueList_currentTrainingImages = featureValueListForAllTrainingImages[dataset:dataset + INCREMENTS]
        actualLabel_currentTrainingImages = actualLabelForTrainingList[dataset:dataset + INCREMENTS]

        print("\n\n\n\n\n Training ON {0} to {1} data".format(dataset, dataset + INCREMENTS))
        ImageFeatureLabelZipList = zip(featureValueList_currentTrainingImages, actualLabel_currentTrainingImages)

        startTimer = time.time()
        ''' ####################  TRAINING PHASE FOR PERCEPTRON ############# '''
        for featureValueListPerImage, actualLabel in ImageFeatureLabelZipList:
            perceptronClassifier.runModel(True, featureValueListPerImage, actualLabel)
        endTimer = time.time()

        perceptron_time.append(endTimer - startTimer)

        startTimer = time.time()
        ''' ####################  TRAINING PHASE FOR NAIVE BYES ############# '''
        naiveBayesClassifier.constructLabelsProbability(actualLabel_currentTrainingImages)
        naiveBayesClassifier.constructFeaturesProbability(featureValueList_currentTrainingImages,
                                                          actualLabel_currentTrainingImages,
                                                          POSSIBLE_VALUES)
        endTimer = time.time()

        bayes_time.append(endTimer - startTimer)

        ''' ################## NO TRAINING PHASE FOR KNN #################  '''
        KNNClassifier.storeTrainingSet(featureValueList_currentTrainingImages, actualLabel_currentTrainingImages)
        ''' SIMPLY STORING FOR KNN '''

        ''' ####################  TESTING PHASE ############# '''
        samples.initTestIters()

        print("TESTING our model that is TRAINED ON {0} to {1} data".format(0, dataset + INCREMENTS))

        perceptron_errorPrediction = naiveByes_errorPrediction = knn_errorPrediction = total = 0
        featureValueListForAllTestingImages, actualTestingLabelList = \
            dataClassifier.extractFeatures(samples.test_lines_itr, samples.test_labelsLines_itr)

        for featureValueListPerImage, actualLabel in zip(featureValueListForAllTestingImages, actualTestingLabelList):
            perceptron_errorPrediction += perceptronClassifier.runModel(False, featureValueListPerImage, actualLabel)
            naiveByes_errorPrediction += naiveBayesClassifier.testModel(featureValueListPerImage, actualLabel)

            ''' ####################  TESTING PHASE FOR KNN ############# '''
            startTimer = time.time()

            knn_errorPrediction += KNNClassifier.test(featureValueListPerImage, actualLabel)

            endTimer = time.time()
            knn_time.append(endTimer - startTimer)
            ''' ####################  TESTING PHASE OVER FOR KNN ############# '''

            total += 1

        perceptron_error = error(perceptron_errorPrediction, total)
        bayes_error = error(naiveByes_errorPrediction, total)
        knn_error = error(knn_errorPrediction, total)
        perceptron_msd.append(perceptron_error)
        bayes_msd.append(bayes_error)
        knn_msd.append(knn_error)

        perceptron_msd_graph = mean_standard_deviation(perceptron_msd,"Perceptron")
        bayes_msd_graph = mean_standard_deviation(bayes_msd,"Bayes")
        knn_msd_graph = mean_standard_deviation(knn_msd,"KNN")

        dataset += INCREMENTS

        dataSetIncrements.append(dataset)
        perceptron_y.append(perceptron_error)
        bayes_y.append(bayes_error)
        knn_y.append(knn_error)

    final_array = {
        1: [perceptron_y, bayes_y, knn_y],
        2: ["Perceptron", "Bayes", "KNN"]
    }

    final_array2 = {
        1: [perceptron_time, bayes_time, knn_time],
        2: ["Perceptron", "Bayes", "KNN"]
    }

    final_array3 = {
        1: [perceptron_msd_graph, bayes_msd_graph, knn_msd_graph],
        2: ["Perceptron", "Bayes", "KNN"]
    }

    error = Error()
    error.graphplot(dataSetIncrements, final_array.get(1), final_array.get(2), inp) #For error plotting
    # error.graphplot(dataSetIncrements, final_array2.get(1), final_array2.get(2), inp) #For time
    error.graphplot(dataSetIncrements, final_array3.get(1), final_array3.get(2), inp) #For mean
    # error.graphplot(dataSetIncrements, final_array3.get(1)[1], final_array3.get(2), inp) #For Standard Deviation

    samples.closeFiles()
