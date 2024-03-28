import pandas as pd
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder


def preprocess(data):
    print("Transforming data")

    return data


train_set = pd.read_csv("./data10", header=None)
print("Train Set shape: ", train_set.shape)

# preprocessing
train_set[1], train_protocols = pd.factorize(train_set[1])
train_set[2], train_services = pd.factorize(train_set[2])
train_set[3], train_flags = pd.factorize(train_set[3])
train_set[41], train_attacks = pd.factorize(train_set[41])

# Splitting the data into features and labels
X_train = train_set.iloc[:, :train_set.shape[1] - 1]  # Features of train set
y_train = train_set.iloc[:, train_set.shape[1] - 1:]  # Label of train set

X_train = pd.DataFrame(X_train)  # dataframe of features
y_train = y_train.values.ravel()  # array of labels


clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10,
                             min_samples_split=2, min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0, max_features=None,
                             random_state=None, max_leaf_nodes=None,
                             min_impurity_decrease=0.0, class_weight=None)

trained_model = clf.fit(X_train, y_train)


print("Score (training set): ", trained_model.score(X_train, y_train))


# TEST
test_set = pd.read_csv("./test10", header=None)
print("Test Set shape:  ", test_set.shape)

test_set[1], test_protocols = pd.factorize(test_set[1])
test_set[2], test_services = pd.factorize(test_set[2])
test_set[3], test_flags = pd.factorize(test_set[3])
test_set[41], test_attacks = pd.factorize(test_set[41])

X_test = test_set.iloc[:, :test_set.shape[1] - 1]  # Features of test set
y_test = test_set.iloc[:, test_set.shape[1] - 1:]  # Labels of test set

X_test = pd.DataFrame(X_test)  # dataframe of features
y_test = y_test.values.ravel()  # array of labels

print("Score (test set): ", trained_model.score(X_test, y_test))
