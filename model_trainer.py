from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score


def train(x_train, y_train):
    # CHOOSE CLASSIFIER (uncomment one of the following)

    # default classifiers:
    clf = RandomForestClassifier()
    # clf = OneVsRestClassifier(Perceptron())  # one classifier for each class (n) [fast, great for large datasets]
    # clf = OneVsOneClassifier(Perceptron())  # one classifier for each pair of classes (n*(n-1)/2) [robust]
    # with custom hyperparameters:
    # clf = OneVsRestClassifier(Perceptron(max_iter=1000, tol=1e-3))
    # clf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, min_samples_split=5)

    print("using: ", clf, "...")
    trained_model = clf.fit(x_train, y_train)
    return trained_model


def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    accuracy = (accuracy_score(y, y_pred)*100)
    return accuracy
