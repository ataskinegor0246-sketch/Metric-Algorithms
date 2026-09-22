import numpy as np

class KMeansClassifier:
    """Класс для кластеризации данных на  основе алгоритма K-means"""
    def __init__(self, centers: int, max_iters: int, floor: float):
        self.centers = centers
        self.max_iters = max_iters
        self.floor = floor

        self.centroids = None
        self.Y = None
        self.inertia_ = None

    def _initCenters(self, X: np.ndarray) -> None:
        """Приватный метод для инициализации центроидов"""
        self.centroids = np.random.choice(X, size = self.centers, replace=False)

    def _inertia(self, X: np.ndarray) -> None:
        """Основная метрика, которую мы будем минимизировать - 
        сумма квадратов расстояний до центроидов"""
        self.inertia_ = np.sum((X - self.centroids[self.Y])**2)

    def _assignCluster(self, X: np.ndarray) -> None:
        """Метод для присваивания объектов к центроидам"""
        distance = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

        self.labels = np.argmin(distance, axis=1)

    
