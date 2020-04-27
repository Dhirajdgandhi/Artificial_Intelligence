import matplotlib.pyplot as plt
from matplotlib import pyplot


class Error:

    def graphplot(self, dataset, errorRateList, type, method):
        for i in range(len(errorRateList)):
            plt.plot(dataset, errorRateList[i], label=type[i])
            plt.xlim(0, dataset[-1] + dataset[-1]/10)
            plt.ylim(0, 100)

        for i in range(len(errorRateList)):
            for data, errorRate in zip(dataset, errorRateList[i]):
                pyplot.text(data, errorRate, str(int(errorRate)))

        plt.title(method)
        plt.legend()
        plt.show()