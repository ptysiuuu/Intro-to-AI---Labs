import numpy as np
from typing import Protocol, Sequence
from dataclasses import dataclass


# Definicja protokołu dla funkcji aktywacji
class ActivationFunc(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray: ...

    def derivative(self, x: np.ndarray) -> np.ndarray: ...


class LossFunc(Protocol):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray): ...

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray): ...


# Implementacje funkcji aktywacji
class ReLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Sigmoid:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self(x)
        return s * (1 - s)


class Tanh:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class MSE(LossFunc):
    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return y_pred - y_true


@dataclass
class GradParams:
    X: np.ndarray
    Y: np.ndarray
    epochs: int
    learning_rate: float
    batch_size: int = 32


@dataclass
class ESParams:
    X: np.ndarray
    Y: np.ndarray
    iterations: int
    sigma: float


@dataclass
class MLPParams:
    layer_sizes: Sequence[int]
    activation_fun: ActivationFunc
    loss_func: LossFunc


# Klasa wielowarstwowego perceptronu
class MLP:
    def __init__(
        self,
        params: MLPParams,
    ):
        layer_sizes = params.layer_sizes

        self.layer_sizes = params.layer_sizes
        self.layers_num = len(layer_sizes)
        self.activation_fun = params.activation_fun
        self.activation_derivative = params.activation_fun.derivative
        self.loss_fun = params.loss_func
        self.loss_derivative = params.loss_func.derivative

        # Inicjalizacja wag i biasów
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            for i in range(self.layers_num - 1)
        ]
        self.biases = [
            np.zeros((1, layer_sizes[i + 1])) for i in range(self.layers_num - 1)
        ]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Propagacja wprzód"""
        self.a = [X]  # Lista aktywacji dla każdej warstwy
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.a[-1], w) + b  # Obliczanie sumy wag i biasu
            self.a.append(self.activation_fun(z))  # Zastosowanie funkcji aktywacji
        return self.a[-1]  # Zwrócenie wyjścia z ostatniej warstwy

    def backward(self, Y: np.ndarray, learning_rate: float = 0.01):
        """Propagacja wsteczna (backpropagation)"""
        # Obliczanie błędu w warstwie wyjściowej
        deltas = [
            self.loss_derivative(Y, self.a[-1]) * self.activation_derivative(self.a[-1])
        ]

        # Propagacja błędu przez warstwy ukryte
        for i in range(self.layers_num - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(
                self.a[i]
            )
            deltas.insert(0, delta)  # Dodanie delty na początek listy

        # Aktualizacja wag i biasów
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.a[i].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train_gradient(self, params: GradParams):
        X, Y, epochs, learning_rate, batch_size = params
        num_samples = len(X)
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            x_shuffled = X[indices]
            y_shuffled = Y[indices]
            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                x_batch = x_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                y_pred_batch = self.forward(x_batch)
                batch_loss = self.loss_fun(y_batch, y_pred_batch)
                epoch_loss += batch_loss
                self.backward(y_batch, learning_rate)
            epoch_loss /= num_samples / batch_size
            if epoch % 10 == 0:  # Loguj co 10 epok
                print(f"Epoch {epoch}, Loss: {epoch_loss}")

    # Funkcje pomocnicze dla algorytmu ewolucyjnego
    def _set_weights(self, flat_weights: np.ndarray):
        start = 0
        for i in range(len(self.weights)):
            weight_shape = self.weights[i].shape
            bias_shape = self.biases[i].shape

            weight_size = np.prod(weight_shape)
            bias_size = np.prod(bias_shape)

            self.weights[i] = flat_weights[start : start + weight_size].reshape(
                weight_shape
            )
            start += weight_size

            self.biases[i] = flat_weights[start : start + bias_size].reshape(bias_shape)
            start += bias_size

    def _get_weights(self):
        flat_weights = []
        for w, b in zip(self.weights, self.biases):
            flat_weights.extend(w.flatten())
            flat_weights.extend(b.flatten())
        return np.array(flat_weights)

    def train_es(self, params: ESParams):
        X, y, iterations, sigma = params
        weights = self._get_weights()
        dim = len(weights)
        success_count = 0
        success_threshold = 1 / 5

        for i in range(iterations):
            mutation = np.random.randn(dim) * sigma
            new_weights = weights + mutation

            self._set_weights(weights)
            predictions = self.forward(X)
            loss = self.loss_fun(y, predictions)

            self._set_weights(new_weights)
            predictions = self.forward(X)
            new_loss = self.loss_fun(y, predictions)

            if new_loss < loss:
                weights = new_weights
                success_count += 1
            else:
                self._set_weights(weights)

            if (i + 1) % 10 == 0:
                success_rate = success_count / 10
                if success_rate > success_threshold:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                success_count = 0

            if (i + 1) % 10 == 0:
                print(f"Iteracja {i+1}: Loss = {loss:.4f}, Sigma = {sigma:.4f}")

        self._set_weights(weights)
