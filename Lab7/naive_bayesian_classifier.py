from typing import Protocol
import numpy as np


class PDFProtocol(Protocol):
    def __call__(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        pass


class GaussianPDF(PDFProtocol):
    def __call__(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        epsilon = 1e-6
        coeff = 1.0 / np.sqrt(2.0 * np.pi * (var + epsilon))
        exponent = -((x - mean) ** 2) / (2.0 * (var + epsilon))
        return coeff * np.exp(exponent)


class NaiveBayes:
    def __init__(self, pdf: PDFProtocol = None):
        self.class_priors = {}
        self.feature_stats = {}
        self.classes = []
        self.pdf = pdf if pdf is not None else GaussianPDF()

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        self.class_priors = {cls: np.sum(y == cls) / n_samples for cls in self.classes}

        self.feature_stats = {}
        for cls in self.classes:
            class_mask = (y == cls)
            X_cls = X[class_mask]
            self.feature_stats[cls] = {
                "mean": np.mean(X_cls, axis=0),
                "var": np.var(X_cls, axis=0),
            }

    def predict(self, X: np.ndarray):
        n_classes = len(self.classes)
        n_samples = X.shape[0]

        scores = np.zeros((n_classes, n_samples))

        for idx, cls in enumerate(self.classes):
            mean = self.feature_stats[cls]["mean"]
            var = self.feature_stats[cls]["var"]
            log_prior = np.log(self.class_priors[cls])

            log_likelihood = np.sum(np.log(self.pdf(X, mean, var) + 1e-9), axis=1)

            scores[idx, :] = log_prior + log_likelihood

        predicted_class_indices = np.argmax(scores, axis=0)
        predictions = [self.classes[idx] for idx in predicted_class_indices]

        return predictions


    # For compatibility with Sklearn
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
