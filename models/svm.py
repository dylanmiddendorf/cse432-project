from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from utils import train_test_dataframes

def main():
    train, test = train_test_dataframes("./datasets")
    
    # Subsample the training dataset to reduce computational overhead
    train = train.sample(10_000)
    
    X_train, y_train = train.drop("label", axis=1), train["label"]
    X_test, y_test = test.drop("label", axis=1), test["label"]

    clf = SVC(cache_size=60000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
