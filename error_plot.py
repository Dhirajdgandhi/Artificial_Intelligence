import matplotlib.pyplot as plt
from matplotlib import pyplot


class Error:

    def __init__(self, classifier, dataSetIncrements, dataset):
        self.classifier = classifier
        self.dataSetIncrements = dataSetIncrements
        self.dataset = dataset

    def graphplot(self, errorRateList, yLabel):
        # for i in range(len(errorRateList)):
        # for i in range(len(errorRateList)):
        for i in range(len(errorRateList)):
            plt.plot(self.dataSetIncrements, errorRateList[i])
        plt.xlabel("Dataset size")
        plt.ylabel(yLabel)
        # plt.xlim(0, self.dataSetIncrements[-1] + self.dataSetIncrements[-1] / 10)
        # plt.ylim(0, 100)

        # for data, errorRate in zip(self.dataSetIncrements, errorRateList):
        #     pyplot.text(data, errorRate, str(errorRate))

        plt.title(self.classifier)
        plt.show();

    def meansd_graphplot(self, errorRateList, yLabel):
        plt.plot(self.dataSetIncrements, errorRateList)
        plt.xlabel("Dataset size")
        plt.ylabel(yLabel)
        # plt.xlim(0, self.dataSetIncrements[-1] + self.dataSetIncrements[-1] / 10)
        # plt.ylim(0, 100)

        for data, errorRate in zip(self.dataSetIncrements, errorRateList):
            pyplot.text(data, errorRate, str(errorRate))

        plt.title(self.classifier)
        plt.show();