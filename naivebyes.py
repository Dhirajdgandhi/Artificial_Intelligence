import numpy as np

# Smotthing
kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]

class NaiveBayesClassifier:

    P_A_GIVEN_B = 'P(A|B)'
    P_B_GIVEN_A = 'P(B|A)'
    P_A = 'P(A)'
    P_B = 'P(B)'

    def __init__(self, FEATURES, LABELS, POSSIBLE_VALUES):
        self.LabelMap = {}
        self.FeatureMap = {}
        self.FEATURES = FEATURES
        self.LABELS = LABELS

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

        # Calculating probability -> frequency/total
        for key in self.LabelMap:
            self.LabelMap[key] = self.LabelMap[key] / totalDataset

    def constructFeaturesProbability(self, featureValueListForAllTrainingImages, actualLabelForTrainingList, POSSIBLE_VALUES):

        # TRAINING
        for label, featureValuesPerImage in zip(actualLabelForTrainingList, featureValueListForAllTrainingImages):
            for feature in range(0, self.FEATURES):
                self.FeatureMap[feature][label][featureValuesPerImage[feature]] += 1

        # Converting frequencies to probabilities
        for featureIndex in range(self.FEATURES):
            for labelIndex in range(self.LABELS):
                sum = 0
                for possibleValueIndex in POSSIBLE_VALUES:
                    sum += self.FeatureMap.get(featureIndex).get(labelIndex).get(possibleValueIndex) + 5
                for possibleValueIndex in POSSIBLE_VALUES:
                    self.FeatureMap[featureIndex][labelIndex][possibleValueIndex] = \
                        self.FeatureMap.get(featureIndex).get(labelIndex).get(possibleValueIndex) + 5 / sum

        # print(self.FeatureMap)

    def predictLabel_GivenFeatures(self, featuresListOfImage):
        probabilityPerLabel = []
        for label in self.LabelMap:
            # P(Y=label|features)
            P_Y = self.LabelMap.get(label)
            P_features_given_Y = 1
            for feature in range(0, self.FEATURES):
                P_features_given_Y = P_features_given_Y*self.FeatureMap[feature][label][featuresListOfImage[feature]]
            probability = P_Y * P_features_given_Y
            probabilityPerLabel.append(probability)

        predictedLabel = np.argmax(probabilityPerLabel)
        return predictedLabel

    def testModel(self, featuresListOfImage, actualLabel):
        predictedLabel = self.predictLabel_GivenFeatures(featuresListOfImage)
        if predictedLabel != actualLabel:
            return 1
        else:
            return 0
