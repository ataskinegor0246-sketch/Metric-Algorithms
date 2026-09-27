import numpy as np

class KMeansClassifier:
    """Класс для кластеризации данных на  основе алгоритма K-means"""
    def __init__(self, k: int, max_iters: int, threshhold: float):
        self.k = k
        self.max_iters = max_iters
        self.threshhold = threshhold

        self.centroids = None
        self.labels = None
        #список со значениями метрики для минимизации
        self.inertia_ = None

    def _init_centers(self, X: np.ndarray) -> None:
        """Метод для инициализации центроидов"""
        self.centroids = np.random.choice(X, size = self.k, replace=False)

    def _inertia(self, X: np.ndarray) -> float:
        """Основная метрика, которую мы будем минимизировать - 
        сумма квадратов расстояний до центроидов"""
        return np.sum((X - self.centroids[self.labels])**2)

    def _assign_cluster(self, X: np.ndarray) -> None:
        """Метод для присваивания объектов к центроидам"""
        distance = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

        self.labels = np.argmin(distance, axis=1)
        
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Метод для обновления центроидов"""
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else self.centroids[k]
            for k in range(self.k)
        ])
        return new_centroids
    
    def fit(self, X: np.ndarray) -> None:
        """Метод обучения"""
        #инициализируем центры и кластеры
        self._init_centers(X)
        self._assign_cluster(X)
        
        #список с изменением метрик
        self.inertia_ = [self._inertia(X)]
        
        for _ in range(self.max_iters):
            new_centroids = self._update_centroids(X, self.labels)
            
            cord_diff = np.abs(new_centroids - self.centroids)
            
            if cord_diff.mean(axis=0) <= self.threshhold:
                break
            
        
            
        
        
        
        
        
        
