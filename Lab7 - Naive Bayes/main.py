import numpy as np
from naive_bayesian_classifier import *
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def make_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.title("Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.show()


def compare_classifiers(classifiers, X, y):
    accuracy_scores = []
    kf = KFold(n_splits=5)
    for clf_name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
        accuracy_scores.append(np.mean(scores))

    plt.figure(figsize=(10, 6))
    plt.scatter(classifiers.keys(), accuracy_scores, color='red', s=100, edgecolor='black', zorder=5)
    plt.xlabel('Classifier', fontweight="bold")
    plt.ylabel('Average accuracy', fontweight="bold")
    plt.title('Comparison of average accuracy for different classifiers', fontweight="bold")
    plt.ylim([0.85, 1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    iris = fetch_ucirepo(id=53)
    X, y = np.array(iris.data.features), np.array(iris.data.targets)
    y = np.ravel(y)

    clf = NaiveBayes()

    classifiers = {
        'Naive Bayes': clf,
        'k-NN': KNeighborsClassifier(),
        'SVC': SVC(kernel='rbf'),
        'Decision Tree': DecisionTreeClassifier()
    }

    compare_classifiers(classifiers, X, y)
