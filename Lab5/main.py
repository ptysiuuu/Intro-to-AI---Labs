import numpy as np
from data_generator import DataGenerator
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib

matplotlib.use("Agg")  # Użycie backendu nieinteraktywnego
from multilayer_perceptron import (
    MLP,
    ActivationFunc,
    LossFunc,
    ReLU,
    Sigmoid,
    Tanh,
    MSE,
    GradParams,
    ESParams,
    MLPParams,
)
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass


@dataclass
class TestParams:
    domain: tuple
    dim: int
    samples_density: int
    test_size: int
    activation_fun: ActivationFunc
    loss_fun: LossFunc
    epochs: int
    learning_rate: float


def investigated_function(x_vec):
    """Badana funkcja"""
    x = x_vec[:, 0]
    result = x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)
    return result.reshape(-1, 1)


def plot_results(
    x_test, y_test, y_pred, domain, layer_sizes, title="Model vs. Investigated Function"
):
    """Funkcja rysująca wykres porównawczy i zapisująca go do pliku"""
    plt.figure(figsize=(10, 6))

    # Rysowanie badanej funkcji
    x_domain = np.linspace(domain[0], domain[1], 500).reshape(-1, 1)
    y_investigated = investigated_function(x_domain)
    plt.plot(
        x_domain,
        y_investigated,
        label="Investigated Function",
        color="blue",
        linewidth=2,
    )

    # Rysowanie punktów testowych
    plt.scatter(x_test, y_test, label="Test Points", color="green", alpha=0.7)

    # Rysowanie przewidywań modelu
    plt.scatter(x_test, y_pred, label="Model Predictions", color="red", alpha=0.7)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Generowanie nazwy pliku na podstawie rozmiarów warstw
    filename = f"mlp_plot_layers_{'_'.join(map(str, layer_sizes))}.png"
    plt.savefig(filename)
    plt.close()  # Zamknięcie wykresu, aby uniknąć problemów z pamięcią
    print(f"Plot saved as {filename}")


def test_MLP(layer_sizes_list, params):
    """Testowanie modelu MLP"""
    gen = DataGenerator()
    x_train, y_train = gen.get_train_data(
        params.domain, params.dim, investigated_function, params.samples_density
    )
    x_test, y_test = gen.get_test_data(
        params.domain, params.dim, investigated_function, params.test_size
    )

    # Skalowanie danych
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaler.fit(x_train)
    y_scaler.fit(y_train)

    norm_x_train = x_scaler.transform(x_train)
    norm_x_test = x_scaler.transform(x_test)
    norm_y_train = y_scaler.transform(y_train)
    norm_y_test = y_scaler.transform(y_test)

    # Testowanie dla różnych rozmiarów warstw
    for layer_sizes in layer_sizes_list:
        mlp_params = MLPParams(layer_sizes, params.activation_fun, params.loss_fun)
        mlp = MLP(mlp_params)
        grad_params = GradParams(norm_x_train, norm_y_train, params.epochs, params.learning_rate)
        mlp.train_gradient(grad_params)

        y_pred = mlp.forward(norm_x_test)
        denorm_y_pred = y_scaler.inverse_transform(y_pred)

        accuracy = mse(y_test, denorm_y_pred)
        print(f"Layer sizes: {layer_sizes}, MSE: {accuracy}")
        # Rysowanie wyników i zapisywanie wykresu
        plot_results(
            x_test,
            y_test,
            denorm_y_pred,
            params.domain,
            layer_sizes,
            title=f"Layer sizes: {layer_sizes}",
        )


def plot_mse_results(results, title, filename, layers=False):
    """Funkcja do rysowania wykresu MSE"""
    if layers:
        desc = "Number of Hidden Layers"
        filename = filename.replace(".png", "_layers.png")
    else:
        desc = "Number of Neurons"
        filename = filename.replace(".png", "_neurons.png")
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        layer_sizes, mse_values = zip(*data)
        plt.plot(layer_sizes, mse_values, label=method)

    plt.title(title, fontweight="bold")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(desc, fontweight="bold")
    plt.ylabel("Mean Squared Error", fontweight="bold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"MSE plot saved as {filename}")


def test_mlp_with_different_neurons(params, method="gradient", n_trials=5):
    """Testowanie wpływu liczby neuronów na jakość aproksymacji"""
    gen = DataGenerator()
    x_train, y_train = gen.get_train_data(
        params.domain, params.dim, investigated_function, params.samples_density
    )
    x_test, y_test = gen.get_test_data(
        params.domain, params.dim, investigated_function, params.test_size
    )

    # Skalowanie danych
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaler.fit(x_train)
    y_scaler.fit(y_train)

    norm_x_train = x_scaler.transform(x_train)
    norm_x_test = x_scaler.transform(x_test)
    norm_y_train = y_scaler.transform(y_train)
    norm_y_test = y_scaler.transform(y_test)

    results = []
    for neurons in range(5, 55, 5):  # Testowanie dla liczby neuronów od 5 do 50
        mse_values = []
        for _ in range(n_trials):
            layer_sizes = [1, neurons, neurons, neurons, 1]
            mlp_params = MLPParams(layer_sizes, params.activation_fun, params.loss_fun)
            mlp = MLP(mlp_params)

            if method == "gradient":
                grad_params = GradParams(norm_x_train, norm_y_train, params.epochs, params.learning_rate)
                mlp.train_gradient(grad_params)
            elif method == "evolutionary":
                es_params = ESParams(norm_x_train, norm_y_train, params.epochs, params.learning_rate)
                mlp.train_es(es_params)
            else:
                raise ValueError("Invalid method. Use 'gradient' or 'evolutionary'.")

            y_pred = mlp.forward(norm_x_test)
            denorm_y_pred = y_scaler.inverse_transform(y_pred)

            accuracy = mse(norm_y_test, y_pred)
            mse_values.append(accuracy)

        average_mse = np.mean(mse_values).item()
        results.append((neurons, average_mse))

    return results


def test_mlp_with_different_layers(params, method="gradient", n_trials=5):
    """Testowanie wpływu liczby warstw na jakość aproksymacji (po 20 neuronów w każdej warstwie)"""
    gen = DataGenerator()
    x_train, y_train = gen.get_train_data(
        params.domain, params.dim, investigated_function, params.samples_density
    )
    x_test, y_test = gen.get_test_data(
        params.domain, params.dim, investigated_function, params.test_size
    )

    # Skalowanie danych
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaler.fit(x_train)
    y_scaler.fit(y_train)

    norm_x_train = x_scaler.transform(x_train)
    norm_x_test = x_scaler.transform(x_test)
    norm_y_train = y_scaler.transform(y_train)
    norm_y_test = y_scaler.transform(y_test)

    results = []

    for layers in range(1, 6):  # Testowanie dla liczby warstw od 1 do 5
        mse_values = []
        for _ in range(n_trials):  # Wykonywanie kilku prób dla uśrednienia wyników
            layer_sizes = [1] + [20] * layers + [1]  # 20 neuronów w każdej warstwie
            mlp_params = MLPParams(layer_sizes, params.activation_fun, params.loss_fun)
            mlp = MLP(mlp_params)

            if method == "gradient":
                grad_params = GradParams(norm_x_train, norm_y_train, params.epochs, params.learning_rate)
                mlp.train_gradient(grad_params)
            elif method == "evolutionary":
                es_params = ESParams(norm_x_train, norm_y_train, params.epochs, params.learning_rate)
                mlp.train_es(es_params)
            else:
                raise ValueError("Invalid method. Use 'gradient' or 'evolutionary'.")

            y_pred = mlp.forward(norm_x_test)
            denorm_y_pred = y_scaler.inverse_transform(y_pred)

            accuracy = mse(norm_y_test, y_pred)
            mse_values.append(accuracy)

        # Średnia MSE dla danej liczby warstw
        average_mse = np.mean(mse_values).item()
        results.append((layers, average_mse))

    return results


def plot_combined_results(x_test, y_test, predictions, layer_sizes, filename):
    """Generuje wykres przedstawiający dane testowe, przewidywania modelu i badaną funkcję."""
    plt.figure(figsize=(12, 6))
    x_domain = np.linspace(-10, 10, 500).reshape(-1, 1)
    y_investigated = investigated_function(x_domain)
    plt.plot(
        x_domain,
        y_investigated,
        label="Investigated Function",
        color="blue",
        linewidth=2,
    )
    plt.scatter(x_test, y_test, label="Test Points", color="green")
    plt.scatter(x_test, predictions, label="Model Predictions", color="red")
    plt.title(f"Layer sizes: {layer_sizes}", fontweight="bold")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Combined plot saved as {filename}")


def test_mlp_with_different_layers_and_neurons(params, n_trials=5):
    """Testowanie wpływu różnych kombinacji liczby warstw i neuronów na jakość aproksymacji."""
    gen = DataGenerator()
    x_train, y_train = gen.get_train_data(
        params.domain, params.dim, investigated_function, params.samples_density
    )
    x_test, y_test = gen.get_test_data(
        params.domain, params.dim, investigated_function, params.test_size
    )

    # Skalowanie danych
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaler.fit(x_train)
    y_scaler.fit(y_train)

    norm_x_train = x_scaler.transform(x_train)
    norm_x_test = x_scaler.transform(x_test)
    norm_y_train = y_scaler.transform(y_train)
    norm_y_test = y_scaler.transform(y_test)

    layer_configurations = [
        [1, 5, 1],
        [1, 10, 1],
        [1, 10, 10, 1],
        [1, 10, 10, 10, 1],
        [1, 20, 20, 1],
        [1, 20, 20, 20, 1],
    ]

    results = []
    for layer_sizes in layer_configurations:
        mse_values = []
        for _ in range(n_trials):
            mlp_params = MLPParams(layer_sizes, params.activation_fun, params.loss_fun)
            mlp = MLP(mlp_params)
            grad_params = GradParams(norm_x_train, norm_y_train, params.epochs, params.learning_rate)
            mlp.train_gradient(grad_params)

            y_pred = mlp.forward(norm_x_test)
            denorm_y_pred = y_scaler.inverse_transform(y_pred)

            accuracy = mse(y_test, denorm_y_pred)
            mse_values.append(accuracy)

        average_mse = np.mean(mse_values)
        results.append((layer_sizes, average_mse))

        # Plot combined results for the last trial
        plot_combined_results(x_test, y_test, denorm_y_pred, layer_sizes, f"combined_plot_{layer_sizes}.png")

    return results


def test_mlp_with_different_activations(params, n_trials=5):
    """Testowanie wpływu różnych funkcji aktywacji na jakość aproksymacji."""
    gen = DataGenerator()
    x_train, y_train = gen.get_train_data(
        params.domain, params.dim, investigated_function, params.samples_density
    )
    x_test, y_test = gen.get_test_data(
        params.domain, params.dim, investigated_function, params.test_size
    )

    # Skalowanie danych
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaler.fit(x_train)
    y_scaler.fit(y_train)

    norm_x_train = x_scaler.transform(x_train)
    norm_x_test = x_scaler.transform(x_test)
    norm_y_train = y_scaler.transform(y_train)
    norm_y_test = y_scaler.transform(y_test)

    layer_configurations = [
        [1, 10, 1],
        [1, 10, 10, 1],
        [1, 10, 10, 10, 1],
        [1, 20, 20, 1],
        [1, 20, 20, 20, 1],
    ]

    activation_functions = {"ReLU": ReLU(), "Sigmoid": Sigmoid(), "Tanh": Tanh()}

    results = {name: [] for name in activation_functions.keys()}

    for name, activation_fun in activation_functions.items():
        for layer_sizes in layer_configurations:
            mse_values = []
            for _ in range(n_trials):
                mlp_params = MLPParams(layer_sizes, activation_fun, params.loss_fun)
                mlp = MLP(mlp_params)
                grad_params = GradParams(norm_x_train, norm_y_train, params.epochs, params.learning_rate)
                mlp.train_gradient(grad_params)

                y_pred = mlp.forward(norm_x_test)
                denorm_y_pred = y_scaler.inverse_transform(y_pred)

                accuracy = mse(y_test, denorm_y_pred)
                mse_values.append(accuracy)

            average_mse = np.mean(mse_values).item()
            results[name].append((layer_sizes, average_mse))

    plot_activation_results(
        results,
        "Impact of Activation Functions on MLP Accuracy",
        "activation_comparison_mse.png",
    )


def plot_activation_results(results, title, filename):
    """Funkcja do rysowania wykresu MSE dla różnych funkcji aktywacji"""
    plt.figure(figsize=(10, 6))

    for activation, data in results.items():
        layer_sizes, mse_values = zip(*data)
        layer_sizes_str = ["-".join(map(str, sizes)) for sizes in layer_sizes]
        plt.plot(layer_sizes_str, mse_values, label=activation)

    plt.title(title, fontweight="bold")
    plt.xlabel("MLP Structures", fontweight="bold")
    plt.ylabel("Mean Squared Error", fontweight="bold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"MSE plot saved as {filename}")


if __name__ == "__main__":
    # Parametry testowe
    test_params = TestParams(
        domain=(-10, 10),
        dim=1,
        samples_density=100,
        test_size=100,
        activation_fun=Tanh(),
        loss_fun=MSE(),
        epochs=10000,  # Liczba epok
        learning_rate=0.001,  # Współczynnik uczenia
    )

    # Testowanie wpływu liczby neuronów
    # evolutionary_results = test_mlp_with_different_neurons(test_params, method="evolutionary")
    # gradient_results = test_mlp_with_different_neurons(test_params)

    # evolutionary_results = test_mlp_with_different_layers(test_params, method="evolutionary")
    # gradient_results = test_mlp_with_different_layers(test_params)

    # Przygotowanie danych do wykresu
    # results = {
    #      "Gradient Descent": gradient_results,
    #      "Evolutionary Strategy": evolutionary_results
    # }

    # Rysowanie wykresów
    # plot_mse_results(results, "Comparison of Approximation Quality", "comparison_mse.png", layers=True)
    # test_mlp_with_different_layers_and_neurons(test_params)

    # evolutionary_results = test_mlp_with_different_neurons(test_params, method="evolutionary")
    # gradient_results = test_mlp_with_different_neurons(test_params)
    # results = {
    #      "Gradient Descent": gradient_results,
    #      "Evolutionary Strategy": evolutionary_results
    # }
    # plot_mse_results(results, "Comparison of Approximation Quality", "comparison_mse.png")
    test_mlp_with_different_activations(test_params)
