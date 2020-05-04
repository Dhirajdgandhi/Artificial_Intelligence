import argparse
import math
import time

from error_plot import Error
from knn import KNN
from naivebyes import NaiveBayesClassifier
from perceptron import PerceptronClassifier
from samples import Samples
from utility import mean_standard_deviation


class DataClassifier:
    def __init__(self, imgHeight, imgWidth, LABELS, pixelChars, pixelGrid):
        if pixelChars is None:
            pixelChars = ['#', '+']
        self.pixelGrid = pixelGrid
        self.imgHeight = imgHeight
        self.imgWidth = imgWidth
        self.FEATURES = math.ceil((imgHeight - self.pixelGrid + 1) * (imgWidth - self.pixelGrid + 1)) \
                        # + (self.imgHeight * self.imgWidth)
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
                # featureValueListPerImage.extend(self.splitImageLineFeaturesIntoGridFeatures(imageLinesList, 1))

                totalImages += 1
                actualLabel = labelsLines_itr.__next__()

                featureValueListForAllTestingImages.append(featureValueListPerImage)
                actualLabelList.append(int(actualLabel))

        except StopIteration:
            # print("End of File")
            pass

        return featureValueListForAllTestingImages, actualLabelList


FACE = "FACE"
DIGIT = "DIGIT"
DIR = "DIR"
HEIGHT = "HEIGHT"
WIDTH = "WIDTH"
LABEL = "LABEL"
PIXELS = "PIXELS"
PERCEPTRON = "PERCEPTRON"
NAIVEBAYES = "NAIVEBAYES"
KNN_ = "KNN"
# Smoothing
smoothingValueList = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]

if __name__ == '__main__':

    ''' #### Initialization #### '''
    errorRateList = []
    dataSetIncrements = []
    timeList = []
    meanList = []
    sdList = []

    ''' #### Arguments Parser #### '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--gridSize')
    parser.add_argument('--smoothingValue')
    parser.add_argument('--classifier')
    parser.add_argument('--percentIncrement')
    args = parser.parse_args()

    inp = args.input
    gridSize = int(args.gridSize)
    smoothingValue = float(0.001) # Default Smoothing Value
    if args.smoothingValue is not None: smoothingValue = float(args.smoothingValue)
    percentIncrement = int(args.percentIncrement)
    '''############ '''

    ''' #### Which Classifier #### '''
    classifier = args.classifier

    isPerceptron = False
    isNaiveBayes = False
    isKnn = False
    All = False
    if classifier == PERCEPTRON:
        isPerceptron = True
    elif classifier == NAIVEBAYES:
        isNaiveBayes = True
    elif classifier == KNN_:
        isKnn = True
    else:
        All = True
    ''' ####  #### '''

    possible_featureValues = [x for x in range(0, gridSize * gridSize + 1)]

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
    samples.readFiles()

    ''' Extracting Features from the Training Data '''
    dataset = 0
    featureValueListForAllTrainingImages, actualLabelForTrainingList = \
        dataClassifier.extractFeatures(samples.train_lines_itr, samples.train_labelsLines_itr)

    totalDataset = len(actualLabelForTrainingList)
    increments = int(totalDataset * percentIncrement / 100)

    ''' Initialization of Classifiers '''
    if isPerceptron: perceptronClassifier = PerceptronClassifier(dataClassifier.FEATURES, dataClassifier.LABELS)
    elif isNaiveBayes: naiveBayesClassifier = NaiveBayesClassifier(dataClassifier.FEATURES, dataClassifier.LABELS, possible_featureValues,
                                                                   smoothingValue)
    elif isKnn: KNNClassifier = KNN(num_neighbors=20)
    else: 
        perceptronClassifier = PerceptronClassifier(dataClassifier.FEATURES, dataClassifier.LABELS)
        naiveBayesClassifier = NaiveBayesClassifier(dataClassifier.FEATURES, dataClassifier.LABELS,
                                                    possible_featureValues,
                                                    smoothingValue)
        KNNClassifier = KNN(num_neighbors=20)

    featureValueListForAllTestingImages = actualTestingLabelList = []

    # print("##### TRAINING OUR MODEL ######")
    errorPrediction = total = 0
    while dataset < totalDataset:

        featureValueList_currentTrainingImages = featureValueListForAllTrainingImages[dataset:dataset + increments]
        actualLabel_currentTrainingImages = actualLabelForTrainingList[dataset:dataset + increments]

        # print("\n\n\n\n\n Training ON {0} to {1} data".format(dataset, dataset + increments))
        ImageFeatureLabelZipList = zip(featureValueList_currentTrainingImages, actualLabel_currentTrainingImages)

        ''' ####################  TRAINING PHASE FOR PERCEPTRON ############# '''
        if isPerceptron or All:
            startTimer = time.time()
            for featureValueListPerImage, actualLabel in ImageFeatureLabelZipList:
                perceptronClassifier.runModel(True, featureValueListPerImage, actualLabel)
            endTimer = time.time()
        ''' ####################  TRAINING PHASE FOR NAIVE BYES ############# '''

        if isNaiveBayes or All:
            startTimer = time.time()
            naiveBayesClassifier.constructLabelsProbability(actualLabel_currentTrainingImages)
            naiveBayesClassifier.constructFeaturesProbability(featureValueList_currentTrainingImages,
                                                              actualLabel_currentTrainingImages,
                                                              possible_featureValues)
            # naiveBayesClassifier.oddsratio(actualLabel_currentTrainingImages)
            endTimer = time.time()
        ''' ################## NO TRAINING PHASE FOR KNN #################  '''

        if isKnn or All:
            KNNClassifier.storeTrainingSet(featureValueList_currentTrainingImages, actualLabel_currentTrainingImages)
        ''' SIMPLY STORING FOR KNN '''

        ''' ####################  TESTING PHASE ############# '''
        samples.initTestIters()

        # print("TESTING our model that is TRAINED ON {0} to {1} data".format(0, dataset + increments))

        featureValueListForAllTestingImages, actualTestingLabelList = \
            dataClassifier.extractFeatures(samples.test_lines_itr, samples.test_labelsLines_itr)

        for featureValueListPerImage, actualLabel in zip(featureValueListForAllTestingImages, actualTestingLabelList):
            if isPerceptron: errorPrediction += perceptronClassifier.runModel(False, featureValueListPerImage, actualLabel)
            if isNaiveBayes: errorPrediction += naiveBayesClassifier.testModel(featureValueListPerImage, actualLabel)

            ''' ####################  TESTING PHASE FOR KNN ############# '''
            if isKnn:
                startTimer = time.time()
                errorPrediction += KNNClassifier.test(featureValueListPerImage, actualLabel)
                endTimer = time.time()
            ''' ####################  TESTING PHASE OVER FOR KNN ############# '''

            total += 1

        errorRate = (errorPrediction * 100) / total
        # print("Error is", errorPrediction, "out of Total of ", total, " : ", errorRate)

        # mean, sd = mean_standard_deviation(errorRate, classifier)

        errorRateList.append(int(errorRate))
        timeList.append('%.2f'%(endTimer - startTimer))
        # meanList.append(mean)
        # sdList.append(sd)

        dataset += increments
        dataSetIncrements.append(dataset)

    error = Error(classifier, dataSetIncrements, inp)
    error.graphplot(errorRateList)  # For error plotting
    error.graphplot(timeList) #For time
    # error.graphplot(meanList, inp)  # For mean
    # error.graphplot(sdList, inp) #For Standard Deviation

    samples.closeFiles()
