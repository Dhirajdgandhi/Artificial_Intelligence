import numpy as np

class NaiveBayesClassifier:

    P_A_GIVEN_B = 'P(A|B)'
    P_B_GIVEN_A = 'P(B|A)'
    P_A = 'P(A)'
    P_B = 'P(B)'

    def __init__(self):
        self.LabelMap = {}
        self.FeatureMap = {}
        self.FEATURES = 0
        self.LABELS = 0

    def P_A_given_B(self, map):
        result = ( map.get(NaiveBayesClassifier.P_B_GIVEN_A) * map.get(NaiveBayesClassifier.P_A) )\
                 / map.get(NaiveBayesClassifier.P_B)
        return result

    # Constructing Labels probability
    # PRIOR DISTRIBUTION OVER LABELS #
    def constructLabelsProbability(self, labelsTestImages):
        totalDataset = len(labelsTestImages)

        # Initialization
        for labelIndex in range(0, totalDataset):
            self.LabelMap[labelIndex] = 1

        # Storing Frequency
        for label in labelsTestImages:
            self.LabelMap[label] += 1

        # Calculating probability -> frequency/total
        for key in self.LabelMap:
            self.LabelMap[key] = self.LabelMap[key] / totalDataset

        self.LABELS = len(self.LabelMap)
        print(self.LabelMap)

    def constructFeaturesProbability(self, featureValueListForAllTrainingImages, actualLabelForTrainingList, POSSIBLE_VALUES):
        self.FEATURES = len(featureValueListForAllTrainingImages[0])

        # Initialization of FMAP - FEATURES X LABELS X POSSIBLE_VALUES
        for featureIndex in range(self.FEATURES):
            self.FeatureMap[featureIndex] = {}
            for labelIndex in range(self.LABELS):
                self.FeatureMap[featureIndex][labelIndex] = {}
                for possibleValueIndex in range(POSSIBLE_VALUES):
                    self.FeatureMap[featureIndex][labelIndex][possibleValueIndex] = 0

        # TRAINING
        for label, featureValuesPerImage in zip(actualLabelForTrainingList, featureValueListForAllTrainingImages):
            for feature in range(0, self.FEATURES):
                self.FeatureMap[feature][label][featureValuesPerImage[feature]] += 1

        # Converting frequencies to probabilities
        for featureIndex in range(self.FEATURES):
            for labelIndex in range(self.LABELS):
                sum = 0
                for possibleValueIndex in range(POSSIBLE_VALUES):
                    sum += self.FeatureMap.get(featureIndex).get(labelIndex).get(possibleValueIndex)
                for possibleValueIndex in range(POSSIBLE_VALUES):
                    self.FeatureMap[featureIndex][labelIndex][possibleValueIndex] = \
                        self.FeatureMap.get(featureIndex).get(labelIndex).get(possibleValueIndex) / sum

        print(self.FeatureMap)

    def predictLabel_GivenFeatures(self, featuresListOfImage, actualLabel):
        probabilityPerLabel = []
        for label in self.LabelMap:
            # P(Y=label|features)
            P_Y = self.LabelMap.get(label)
            P_features_given_Y = 1
            for feature in self.FEATURES:
                P_features_given_Y = P_features_given_Y*self.FeatureMap[feature][label][featuresListOfImage[feature]]
            probability = P_Y * P_features_given_Y
            probabilityPerLabel.append(probability)

        predictedLabel = np.argmax(probabilityPerLabel)
        return predictedLabel

    def trainModel(self):
        self.predictLabel_GivenFeatures([], 1)