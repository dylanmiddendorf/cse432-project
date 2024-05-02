import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import time
from utils import train_test_dataframes


def evaluate_model(clf: KNeighborsClassifier) -> float:
    train, test = train_test_dataframes("./datasets")
    X_train, y_train = train.drop("label", axis=1), train["label"]
    X_test, y_test = test.drop("label", axis=1), test["label"]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)  # Can be used for additional metrics
    return accuracy_score(y_test, y_pred)


def grid_search_cv() -> None:
    train, test = train_test_dataframes("./datasets")
    train = train.sample(20_000)  # Reduce computational overhead

    X_train, y_train = train.drop("label", axis=1), train["label"]
    param_grid = {
        "n_neighbors": [1, 2, 4, 8, 16, 32],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }

    start = time.time()
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, param_grid=param_grid, verbose=2, n_jobs=5)
    # print(cross_val_score(knn, X_train, y_train, n_jobs=5, verbose=1))
    clf.fit(X_train, y_train)

    results_df = pd.DataFrame(clf.cv_results_)
    results_df.to_csv("./results/knn_search.csv")
    print(time.time() - start)


def main():
    clf = KNeighborsClassifier(n_neighbors=8, p=1, weights='distance', n_jobs=5)
    print(evaluate_model(clf))


if __name__ == "__main__":
    main()
