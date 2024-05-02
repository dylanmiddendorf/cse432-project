from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from utils import train_test_dataframes

def main():
    train, test = train_test_dataframes("./datasets")
    X_train, y_train = train.drop("label", axis=1), train["label"]
    X_test, y_test = test.drop("label", axis=1), test["label"]

    clf = KNeighborsClassifier(weights="distance")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
