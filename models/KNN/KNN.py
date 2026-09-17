import numpy as np
from collections import Counter

class KNN_classifier:
    """Класс, реализующий метрический алгоритм 
    классификации KNN(K Nearest Neighbors)"""
    def __init__(self, K_neighbors: int) -> None:
        self.K_neighbors = K_neighbors

        self.X = None
        self.Y = None

        self._Confusion_matrix = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Метод для запоминания обучающей выборки"""
        self.X = X
        self.Y = Y

    def predict(self, objects: np.ndarray) -> np.ndarray:
        """Метод по предсказнию класса объекта"""
        predictions = []
        for x in objects:
            #вектор расстояний от таргета до всех объектов их выборки
            distance = np.linalg.norm(self.X - objects, axis=1)

            indexis = np.argpartition(distance, self.K_neighbors)[:self.K_neighbors]
            neighbors = self.Y[indexis]

            #количество соседей каждого класса
            counts = {}
            for neighbor in neighbors:
                counts[neighbor] = counts.get(neighbor, 0) + 1    

            predictions.append(max(counts, ket = counts.get))

        predict = np.array(predictions)
        return predict


    def confusion_matrix(self, predictions: np.ndarray,
                         target: np.ndarray) -> np.ndarray:
        """Возвращает confusion matrix"""
        self._Confusion_matrix = np.zeros((2,2))
        #TP
        self._Confusion_matrix[0,0] = np.sum(predictions == 1 & target == 1)
        #FP
        self._Confusion_matrix[0,1] = np.sum(predictions == 1 & target == 0)
        #FN
        self._Confusion_matrix[1,0] = np.sum(predictions == 0 & target == 1)
        #TN
        self._Confusion_matrix[1,1] = np.sum(predictions == 0 & target == 0)

        return self._Confusion_matrix

    def accuracy(self) -> float:
        """Метрика accuracy"""
        return (self._Confusion_matrix[1,1] + self._Confusion_matrix[0,0])/(self._Confusion_matrix[1,1] + self._Confusion_matrix[0,0] +
                                                                            self._Confusion_matrix[0,1] + self._Confusion_matrix[0,1])
    def precision(self) -> float:
        """Метрика precision"""
        if (self._Confusion_matrix[0,0] + self._Confusion_matrix[0,1]) == 0:
            return 0
        return (self._Confusion_matrix[0,0])/(self._Confusion_matrix[0,0] + self._Confusion_matrix[0,1])

    def recall(self) -> float:
        """Метрика recall"""
        if (self._Confusion_matrix[0,0] + self._Confusion_matrix[1,0]) == 0:
            return 0
        return (self._Confusion_matrix[0,0])/(self._Confusion_matrix[0,0] + self._Confusion_matrix[1,0])
