import numpy as np
import math
import samples

pixelGrid = 1
imgHeight = 20
imgWidth = 29
LABELS = 10
FEATURES = math.ceil((imgHeight * imgWidth) / pixelGrid)
SHAPE = (LABELS, FEATURES)

weightMatrix = np.empty(SHAPE)
# print(weightMatrix)
pixelChars = ['#', '+']


def scanInput(lines_itr, labelsLines_itr, isTrain):
    imageLine = imageLineItr.__next__()
    label = labelLineItr.__next__()
    incorrect = total = 0
    featureValueList = []

    try:
        while imageLine:
            # Skipping the blank lines
            while imageLine and countPixels(imageLine) == 0:
                imageLine = lines_itr.__next__()

            # Scanning image pixels
            for i in range(0, imgHeight):
                featureValueList.extend(extractFeatures(imageLine, i))
                # print(featureValueList)
                imageLine = lines_itr.__next__()

            if isTrain:
                validateLabel(isTrain, featureValueList, label)
            else:
                incorrect += validateLabel(isTrain, featureValueList, label)
                total += 1

            label = labelsLines_itr.__next__()

            # Re-init the feature score
            featureValueList = []
    except StopIteration:
        print("End of File")
        pass

    return incorrect, total


def validateLabel(isTrain, featureValueList, label):
    featureScoreList = []
    for labelWeights in weightMatrix:
        featureScoreList.append(np.sum(labelWeights * featureValueList))
    predictedLabel = np.argmax(featureScoreList)
    actualLabel = int(label)

    if predictedLabel != actualLabel:
        print(predictedLabel, " ", actualLabel)
        if isTrain:
            updateWeights(predictedLabel, actualLabel, featureValueList)
        else:
            return 1
    else:
        return 0


def countPixels(line):
    count = 0
    if not isinstance(line, list):
        line = list(line)

    for char in line:
        if char in pixelChars:
            count += 1

    return count


def extractFeatures(line, row):
    gridList = []
    featureValueList = []

    for startIndexOfGrid in range(0, len(line), pixelGrid):
        gridList.append(line[startIndexOfGrid:startIndexOfGrid + pixelGrid])

    # col = 0
    for grid in gridList:
        # Count the number of chars in this grid and add the count to respective index of FEATURE
        # indexOfFeature = row + col
        featureValueList.append(countPixels(grid))

    return featureValueList


def updateWeights(predictedLabel, actualLabel, featureValueList):
    weightMatrix[actualLabel, :] = weightMatrix[actualLabel, :] + featureValueList
    weightMatrix[predictedLabel, :] = weightMatrix[actualLabel, :] - featureValueList
    #print(weightMatrix[actualLabel, :])
    #print(weightMatrix[predictedLabel, :])


