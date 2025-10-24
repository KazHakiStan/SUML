import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target


def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train, y_train)
    print("Model training complete")
    return clf


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")


def save_model(clf, filename='./app/model.joblib'):
    joblib.dump(clf, filename)
    print(f"Model saved as {filename}")


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = train_model(X_train, y_train)

    evaluate_model(clf, X_test, y_test)

    save_model(clf)


if __name__ == "__main__":
    main()

