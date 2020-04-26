import numpy as np


class PerceptronClassifier:
    def __init__(self, FEATURES, LABELS):
        self.SHAPE = (LABELS, FEATURES + 1)  # The +1 is for our w0 weight.
        self.weightMatrix = np.zeros(self.SHAPE)

    def updateWeights(self, predictedLabel, actualLabel, featureValueList):
        # print("Updating Weights")
        self.weightMatrix[actualLabel] = self.weightMatrix[actualLabel] + featureValueList
        self.weightMatrix[predictedLabel] = self.weightMatrix[predictedLabel] - featureValueList
        # print(weightMatrix[actualLabel, :])
        # print(weightMatrix[predictedLabel, :])

    def runModel(self, isTrain, featureValueList, actualLabel):
        featureScoreList = []
        for labelWeights in self.weightMatrix:
            featureScoreList.append(np.sum(np.dot(labelWeights, featureValueList)))

        # print("Feature Score List :", featureScoreList)
        predictedLabel = np.argmax(featureScoreList)

        if predictedLabel != actualLabel:
            # print(predictedLabel, " ", actualLabel)
            if isTrain:
                self.updateWeights(predictedLabel, actualLabel, featureValueList)
            else:
                return 1
        else:
            return 0

    def initWeightMatrix(self):
        self.weightMatrix = np.zeros(self.SHAPE)  # Randomized

