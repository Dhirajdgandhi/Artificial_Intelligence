import math
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

    def extractFeatures(self, lines_itr, labelsLines_itr, ITERATIONS):
        try:
            imageLine = lines_itr.__next__()
        except StopIteration:
            return None, None

        totalImages = 0
        featureValueListPerImage = [1]
        featureValueListForAllTestingImages = []
        actualLabelList = []

        try:
            while imageLine and totalImages < ITERATIONS:
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
            return None, None

        return featureValueListForAllTestingImages, actualLabelList


if __name__ == '__main__':
    print("TRAINING OUR MODEL FIRST")
    samples = Samples()
    dataClassifier = DataClassifier()
    perceptronClassifier = PerceptronClassifier(dataClassifier.FEATURES, dataClassifier.LABELS)

    samples.readFiles()
    percentdataset = 0

    while percentdataset < 100:
        featureValueListForAllTrainingImages, actualLabelList = dataClassifier.extractFeatures(samples.train_lines_itr, samples.train_labelsLines_itr, 100)

        if featureValueListForAllTrainingImages is None:
            break

        for featureValueListPerImage, actualLabel in zip(featureValueListForAllTrainingImages, actualLabelList):
            perceptronClassifier.runModel(True, featureValueListPerImage, actualLabel)

        percentdataset += 10
        print("Let's TEST our model that is TRAINED ON ", percentdataset, "%")

        errorPrediction = 0
        total = 0
        while total < 1000:
            featureValueListForAllTestingImages, actualLabelList = dataClassifier.extractFeatures(samples.test_lines_itr, samples.test_labelsLines_itr, 1000)

            if featureValueListForAllTestingImages is None:
                break

            for featureValueListPerImage, actualLabel in zip(featureValueListForAllTestingImages, actualLabelList):
                errorPrediction += perceptronClassifier.runModel(False, featureValueListPerImage, actualLabel)
                total += 1

        samples.initTestIters()

        print("Error is", errorPrediction, "out of Total of ", 1000)
        print((errorPrediction * 100) / 1000, "%")

    samples.closeFiles()
