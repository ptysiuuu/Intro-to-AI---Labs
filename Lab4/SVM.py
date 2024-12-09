import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class SVMParams:
    def __init__(self, C: float, max_iter: int, kernel: Literal['linear', 'rbf'], gamma: float, lr: float) -> None:
        self.C = C
        self.max_iter = max_iter
        self.kernel = kernel
        self.gamma = gamma
        self.lr = lr


class SVM:
    def __init__(self, params: SVMParams) -> None:
        self.params = params
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        self.support_vector_bias = None
    
    def linear_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)

    def rbf_kernel(self, x1: np.ndarray, x2: np.ndarray, gamma: float) -> float:
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

    def compute_kernel(self, X: np.ndarray, kernel: Literal['linear', 'rbf'], gamma: float) -> np.ndarray:
        n_samples, _ = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if kernel == 'linear':
                    K[i, j] = self.linear_kernel(X[i], X[j])
                elif kernel == 'rbf':
                    K[i, j] = self.rbf_kernel(X[i], X[j], gamma)
        return K

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass
