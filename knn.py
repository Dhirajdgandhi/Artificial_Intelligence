from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=15)

    def train(self, x_train_,  y_train):
        self.knn.fit(x_train_, y_train)

    def test(self, x_test, actualLabel):
        y_pred = self.knn.predict(x_test)
        if y_pred == actualLabel:
            return 0
        else:
            return 1