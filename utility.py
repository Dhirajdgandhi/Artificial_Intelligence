import statistics
from collections import Counter


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    from math import sqrt
    return sqrt(distance)


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def mean_standard_deviation(errorRate):
    meanList = []
    sdList = []
    # print(errorRate)
    for datapoints in range(0,len(errorRate[0])):
        eachErrorRate = []
        for run in range(0,len(errorRate)):
            eachErrorRate.append(errorRate[run][datapoints])
        meanList.append('%.2f'%statistics.mean(eachErrorRate))
        sdList.append('%.2f'%statistics.stdev(eachErrorRate))
    return meanList, sdList
