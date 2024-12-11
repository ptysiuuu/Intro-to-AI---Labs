import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from SVM import SVM, SVMParams, Kernel
import seaborn as sns
from matplotlib import pyplot as plt


class LinearKernel(Kernel):
    def __call__(self, x1, x2):
        x1 = np.array(x1, dtype=np.float64)
        x2 = np.array(x2, dtype=np.float64)
        return np.dot(x1, x2)


class RBFKernel(Kernel):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def __call__(self, x1, x2):
        x1 = np.array(x1, dtype=np.float64)
        x2 = np.array(x2, dtype=np.float64)
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)


def load_and_prep_wine_data():
    df = pd.read_csv('./winequality-red.csv', sep=';')
    df2 = pd.read_csv('./winequality-white.csv', sep=';')
    df = pd.concat([df, df2])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    median = df['quality'].median()
    df['quality'] = df['quality'].apply(lambda x: 1 if x > median else -1)
    return df


def main():
    if __name__ == "__main__":
        df = load_and_prep_wine_data()
        X = df.drop('quality', axis=1)
        y = df['quality']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lin_kernel = LinearKernel()
        rbf_kernel = RBFKernel(gamma=0.1)

        params = SVMParams(C=1, kernel=rbf_kernel)
        svm_lin = SVM(params)
        svm_lin.fit(X_train, y_train)
        accuracy, cm, report = svm_lin.evaluate(X_test, y_test)

        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()

        print(f'Accuracy: {accuracy}')
        print(f'Report:\n{report}')


main()
