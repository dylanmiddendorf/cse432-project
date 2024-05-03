import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

from utils import train_test_dataframes


def evaluate_model(clf: SVC) -> float:
    train, test = train_test_dataframes("./datasets")
    train = train.sample(10_000)  # Reduce computational overhead

    X_train, y_train = (train.drop("label", axis=1) / 255.0), train["label"]
    X_test, y_test = (test.drop("label", axis=1) / 255.0), test["label"]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)  # Can be used for additional metrics
    return accuracy_score(y_test, y_pred)


def random_search_cv() -> None:
    train, test = train_test_dataframes("./datasets")
    train = train.sample(10_000)  # Reduce computational overhead

    X_train, y_train = (train.drop("label", axis=1) / 255.0), train["label"]
    parameters = {
        "C": np.logspace(-3, 2, 6),
        "kernel": ["linear", "poly", "rbf"],
        "degree": np.arange(1, 5 + 1, 1).tolist(),
        "gamma": np.logspace(-3, 2, 6),
        "coef0": np.arange(0.0, 10.0, 0.1).tolist(),
        "tol": np.arange(0.001, 0.01 + 0.001, 0.001).tolist(),
        "cache_size": [60_000],
    }

    clf = RandomizedSearchCV(SVC(), param_distributions=parameters, verbose=2, n_jobs=5)
    clf.fit(X_train, y_train)

    results_df = pd.DataFrame(clf.cv_results_)
    results_df.to_csv("./results/svc_search_2.csv")


def main():    
    clf = SVC()
    print(evaluate_model(clf))


if __name__ == "__main__":
    main()
