from utility import euclidean_distance, most_frequent


class KNN:
    def __init__(self, num_neighbors):
        self.num_neighbors = num_neighbors
        self.trainX = []
        self.trainY = []

    def test(self, test_row, actualLabel):
        neighbors = self.get_neighbors(test_row)
        y_pred = most_frequent(neighbors)

        if y_pred == actualLabel:
            return 0
        else:
            return 1

    def storeTrainingSet(self, x, y):
        self.trainX += x
        self.trainY += y

    # Locate the most similar neighbors
    def get_neighbors(self, test_row):
        distances = list()
        zippedTrainingData = zip(self.trainX, self.trainY)

        for train_row in zippedTrainingData:
            features = train_row[0]
            label = train_row[1]
            dist = euclidean_distance(test_row, features)
            distances.append((features, label, dist))

        distances.sort(key=lambda tup: tup[2]) # Sort according to the dist
        neighbors = list()
        for i in range(self.num_neighbors):
            neighbors.append(distances[i][1])
        return neighbors

