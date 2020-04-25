import numpy as np


class KNearestNeighbors(object):
    def __init__(self, k):
        self.k = k

    @staticmethod
    def _eucledian_distance(v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        distance = 0
        for i in range(len(v1)-1):
            distance += (v1[i]-v2[i])**2
            return np.sqrt(distance)

    def predict(self, training_set, testing_set):
        distances = []
        for i in range(len(training_set)):
            dist = self._eucledian_distance(training_set[i][:-1], testing_set)
            distances.append((training_set[i], dist))
        distances.sort(key=lambda x: x[1])

        neighgors = []
        for i in range(self.k):
            neighgors.append(distances[i][0])

        classes = {}
        for i in range(len(neighgors)):
            response = neighgors[i][-1]
            if response in classes:
                classes[response] += 1
            else:
                classes[response] = 1

        sorted_classes = sorted(
            classes.items(), key=lambda x: x[1], reverse=True)
        return sorted_classes[0][0]

    @staticmethod
    def evaluate(y_true, y_predicted):
        number_correct = 0
        for actual, predicted in zip(y_true, y_predicted):
            if actual == predicted:
                number_correct += 1
        return number_correct/len(y_true)
