import math
import numpy as np


class NaiveBayesClassifier:

    P_A_GIVEN_B = 'P(A|B)'
    P_B_GIVEN_A = 'P(B|A)'
    P_A = 'P(A)'
    P_B = 'P(B)'

    # Smotthing
    kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]

    def __init__(self, FEATURES, LABELS, POSSIBLE_VALUES):
        self.LabelMap = {}
        self.FeatureMap = {}
        self.FEATURES = FEATURES
        self.LABELS = LABELS
        self.K = 0.001

        # Initialization of FMAP - FEATURES X LABELS X POSSIBLE_VALUES
        for featureIndex in range(self.FEATURES):
            self.FeatureMap[featureIndex] = {}
            for labelIndex in range(self.LABELS):
                self.FeatureMap[featureIndex][labelIndex] = {}
                for possibleValueIndex in POSSIBLE_VALUES:
                    self.FeatureMap[featureIndex][labelIndex][possibleValueIndex] = 0

        # Initialization
        for labelIndex in range(0, self.LABELS):
            self.LabelMap[labelIndex] = 0

    def P_A_given_B(self, map):
        result = ( map.get(NaiveBayesClassifier.P_B_GIVEN_A) * map.get(NaiveBayesClassifier.P_A) )\
                 / map.get(NaiveBayesClassifier.P_B)
        return result

    # Constructing Labels probability
    # PRIOR DISTRIBUTION OVER LABELS #
    def constructLabelsProbability(self, trainingLabels):
        totalDataset = len(trainingLabels)

        # Storing Frequency
        for label in trainingLabels:
            self.LabelMap[label] += 1

        # Calculating probability -> frequency/total -> LOG
        for key in self.LabelMap:
            probability = self.LabelMap[key] / totalDataset
            self.LabelMap[key] = probability

    def constructFeaturesProbability(self, featureValueListForAllTrainingImages, actualLabelForTrainingList, POSSIBLE_VALUES):

        # TRAINING
        for label, featureValuesPerImage in zip(actualLabelForTrainingList, featureValueListForAllTrainingImages):
            for feature in range(0, self.FEATURES):
                self.FeatureMap[feature][label][featureValuesPerImage[feature]] += 1

        # Converting frequencies to probabilities to it's LOG
        for featureIndex in range(self.FEATURES):
            for labelIndex in range(self.LABELS):
                sum = 0
                for possibleValueIndex in POSSIBLE_VALUES:
                    sum += self.FeatureMap.get(featureIndex).get(labelIndex).get(possibleValueIndex) + self.K
                for possibleValueIndex in POSSIBLE_VALUES:
                    probability = (self.FeatureMap.get(featureIndex).get(labelIndex).get(possibleValueIndex) + self.K) / sum
                    self.FeatureMap[featureIndex][labelIndex][possibleValueIndex] = probability

    def predictLabel_GivenFeatures(self, featuresListOfImage):
        probabilityPerLabel = []
        for label in self.LabelMap:
            # P(Y=label|features)
            P_Y = self.LabelMap.get(label)
            P_features_given_Y = 0
            for feature in range(0, self.FEATURES):
                P_features_given_Y += math.log(self.FeatureMap[feature][label][featuresListOfImage[feature]])
            probability = math.log(P_Y, 2) + P_features_given_Y
            probabilityPerLabel.append(probability)

        predictedLabel = np.argmax(probabilityPerLabel)
        return predictedLabel

    def testModel(self, featuresListOfImage, actualLabel):
        predictedLabel = self.predictLabel_GivenFeatures(featuresListOfImage)
        if predictedLabel != actualLabel:
            return 1
        else:
            return 0
