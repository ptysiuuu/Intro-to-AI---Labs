import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_prep_wine_data():
    df = pd.read_csv('winequality-red.csv', sep=';')
    df2 = pd.read_csv('winequality-white.csv', sep=';')
    df = pd.concat([df, df2])
    median = df['quality'].median()
    df['quality'] = df['quality'].apply(lambda x: 1 if x > median else -1)
    return df


def main():
    if __name__ == "__main__":
        df = load_and_prep_wine_data()
        X = df.drop('quality', axis=1)
        y = df['quality']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
