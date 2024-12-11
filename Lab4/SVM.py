import numpy as np
from dataclasses import dataclass
from typing import Literal, Protocol
from cvxopt import matrix, solvers
from sklearn.metrics import confusion_matrix, classification_report


class Kernel(Protocol):
    def __call__(self, x1, x2) -> float:
        pass


@dataclass
class SVMParams:
    def __init__(self, C: float, kernel: Kernel) -> None:
        self.C = C
        self.kernel = kernel


class SVM:
    def __init__(self, params: SVMParams) -> None:
        self.params = params
        self.alphas = None
        self.b = None

    def _compute_gram_matrix(self, X, y):
        N = len(y)
        Q = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Q[i, j] = y[i] * y[j] * self.params.kernel(X[i], X[j])
        return Q

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        N = len(y)

        P = matrix(self._compute_gram_matrix(X, y))
        q = matrix(-np.ones(N))
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = matrix(np.hstack((np.zeros(N), np.ones(N) * self.params.C)))
        A = matrix(y, (1, N), 'd')
        b = matrix(0.0)

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)

        alphas = np.ravel(solution['x'])
        support_vector_idx = np.where((alphas > 1e-5) & (alphas < self.params.C))[0]
        self.support_vectors_ = X[support_vector_idx]
        self.support_labels_ = y[support_vector_idx]
        self.alphas_ = alphas[support_vector_idx]
        self.b = np.mean(
            self.support_labels_ - np.sum(
                self.alphas_ * self.support_labels_ * 
                np.array([[self.params.kernel(sv, x) for x in self.support_vectors_] for sv in self.support_vectors_]),
                axis=1
            )
        )

    def decision_function(self, X):
        X = np.array(X, dtype=np.float64)

        decision_values = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            decision_values[i] = np.sum(
                self.alphas_ * self.support_labels_ * np.array([self.params.kernel(x, sv) for sv in self.support_vectors_])
            ) + self.b
        return decision_values

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        correct = np.sum(y_pred == y_test)

        cm = confusion_matrix(y_test, y_pred)

        report = classification_report(y_test, y_pred)
        return correct / len(y_test), cm, report
