import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from SVM import SVM, SVMParams, Kernel
from enum import Enum
import seaborn as sns
from matplotlib import pyplot as plt


class LinearKernel(Kernel):
    def __call__(self, x1, x2):
        x1 = np.array(x1, dtype=np.float64)
        x2 = np.array(x2, dtype=np.float64)
        return np.dot(x1, x2)

    def __str__(self):
        return "Linear Kernel"


class RBFKernel(Kernel):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def __call__(self, x1, x2):
        x1 = np.array(x1, dtype=np.float64)
        x2 = np.array(x2, dtype=np.float64)
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def __str__(self):
        return "RBF Kernel"


class PolynomialKernel:
    def __init__(self, coef0=1, degree=3):
        self.coef0 = coef0
        self.degree = degree

    def __call__(self, x1, x2):
        return (np.dot(x1, x2) + self.coef0) ** self.degree


def load_and_prep_wine_data():
    df = pd.read_csv("./winequality-red.csv", sep=";")
    df2 = pd.read_csv("./winequality-white.csv", sep=";")
    df = pd.concat([df, df2])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    return df


def extract_from_data(df: pd.DataFrame, n_samples: int = None):
    if n_samples:
        df = df.sample(n_samples)
    df["quality"] = df["quality"].apply(lambda x: 1 if x > 5 else -1)
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return X, y


def accuracy_test(avg_samples, n_samples):
    class Kernels(Enum):
        rbf_kernel = RBFKernel(gamma=0.1)
        lin_kernel = LinearKernel()

    df = load_and_prep_wine_data()

    c_s = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    c_dict = {}

    for kernel in Kernels:
        print(str(kernel.value))
        for c in c_s:
            print(f"C: {c}")
            c_avg = 0
            for n in range(avg_samples):
                X, y = extract_from_data(df, n_samples)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                params = SVMParams(C=c, kernel=kernel.value)
                clf = SVM(params)
                clf.fit(X_train, y_train)
                eval_return = clf.evaluate(X_test, y_test)

                # sns.heatmap(eval_return.cm, annot=True, fmt='d')
                # plt.xlabel('Predicted label', fontweight='bold')
                # plt.ylabel('True label', fontweight='bold')
                # plt.show()

                print(f"Iter {n + 1} Accuracy: {eval_return.accuracy}")
                c_avg += eval_return.accuracy
            c_avg /= avg_samples
            print(f"Avg: {c_avg}")
            c_dict[c] = c_avg

        plt.scatter(c_dict.keys(), c_dict.values(), label=str(kernel.value))
        plt.title(
            "Average accuracy for SVM in relation to the C parameter", fontweight="bold"
        )
        plt.xlabel("C values", fontweight="bold")
        plt.ylabel("Accuracy", fontweight="bold")
    plt.xticks(
        [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
        [r"$1e-3$", r"$1e-2$", r"$1e-1$", r"$1$", r"$1e1$", r"$1e2$", r"$1e3$"],
    )
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.savefig("svm_accuracy.png")
    plt.show()


def rbf_param_test(avg_samples, n_samples):
    sigmas = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    sigm_dict = {}

    df = load_and_prep_wine_data()

    for s in sigmas:
        print(f"Sigma: {s}")
        s_avg = 0
        for n in range(avg_samples):
            X, y = extract_from_data(df, n_samples)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            rbf_kernel = RBFKernel(gamma=s)
            params = SVMParams(C=1, kernel=rbf_kernel)
            clf = SVM(params)
            clf.fit(X_train, y_train)
            eval_return = clf.evaluate(X_test, y_test)

            # sns.heatmap(eval_return.cm, annot=True, fmt='d')
            # plt.xlabel('Predicted label', fontweight='bold')
            # plt.ylabel('True label', fontweight='bold')
            # plt.show()

            print(f"Iter {n + 1} Accuracy: {eval_return.accuracy}")
            s_avg += eval_return.accuracy
        s_avg /= avg_samples
        print(f"Avg: {s_avg}")
        sigm_dict[s] = s_avg
    plt.scatter(sigm_dict.keys(), sigm_dict.values())
    plt.title("RBF Kernel SVM accuracy in relation to sigma", fontweight="bold")
    plt.xlabel("Sigma values", fontweight="bold")
    plt.ylabel("Accuracy", fontweight="bold")
    plt.xticks(
        [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
        [r"$1e-3$", r"$1e-2$", r"$1e-1$", r"$1$", r"$1e1$", r"$1e2$", r"$1e3$"],
    )
    plt.xscale("log")
    plt.grid(True)
    plt.savefig("rbf_accuracy.png")
    plt.show()


def main():
    if __name__ == "__main__":
        AVG_SAMPLES = 3
        N_SAMPLES = 1000
        accuracy_test(AVG_SAMPLES, N_SAMPLES)


main()
