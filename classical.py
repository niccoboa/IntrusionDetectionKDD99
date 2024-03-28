import numpy as np
import pandas as pd
import numpy
import math
from tabulate import tabulate

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, zero_one_loss

print("Loading data")
traindata = pd.read_csv("data10", header=None)
testdata = pd.read_csv("test10", header=None)

print("Number of different attacks in train: ", traindata[41].unique().size)
print("Number of different attacks in test :  ", testdata[41].unique().size)

# GATHERING
print("Gathering (before classification)")

# Dictionary of attacks and their categories
attacks_categories = {
    'imap.': 'r2l',
    'named.': 'r2l',
    'land.': 'dos',
    'processtable.': 'dos',  #
    'back.': 'dos',
    'pod.': 'dos',
    'neptune.': 'dos',
    'buffer_overflow.': 'u2r',
    'teardrop.': 'dos',
    'apache2.': 'dos',  #
    'xsnoop.': 'r2l',  #
    'saint.': 'probe',  #
    'ftp_write.': 'r2l',
    'multihop.': 'r2l',
    'loadmodule.': 'u2r',
    'ipsweep.': 'probe',
    'nmap.': 'probe',
    'sqlattack.': 'r2l',  #
    'warezclient.': 'r2l',
    'snmpguess.': 'r2l',  #
    'satan.': 'probe',
    'portsweep.': 'probe',
    'xlock.': 'r2l',  #
    'mailbomb.': 'dos',  #
    'udpstorm.': 'dos',  #
    'worm.': 'u2r',  #
    'perl.': 'u2r',
    'snmpgetattack.': 'r2l',  #
    'normal.': 'normal',
    'smurf.': 'dos',
    'mscan.': 'probe',  #
    'warezmaster.': 'r2l',
    'sendmail.': 'r2l',  #
    'rootkit.': 'u2r',
    'guess_passwd.': 'r2l',
    'phf.': 'r2l',
    'spy.': 'u2r',
    'ps.': 'u2r',  #
    'xterm.': 'u2r',  #
    'httptunnel.': 'u2r'  #
}

# Replace attacks with categories (gathering)
traindata[41] = traindata[41].replace(attacks_categories)
testdata[41] = testdata[41].replace(attacks_categories)

# Check if gathering was successful
print("Number of different attacks in train NOW: ", traindata[41].unique().size)
print("Number of different attacks in test NOW:  ", testdata[41].unique().size)

# print traindata[41].unique() and testdata[41].unique() without duplicates
print("Categories: ", set(traindata[41].unique()).union(set(testdata[41].unique())))


# PREPROCESSING
print("Transforming data (preprocessing)")
traindata[1], train_protocols = pd.factorize(traindata[1])
traindata[2], train_services = pd.factorize(traindata[2])
traindata[3], train_flags = pd.factorize(traindata[3])
# traindata[41], train_attacks = pd.factorize(traindata[41])
order = {'normal': 0, 'dos': 1, 'r2l': 2, 'u2r': 3, 'probe': 4}
traindata[41] = traindata[41].map(order)
train_attacks = np.array(list(order.keys()))

testdata[1], test_protocols = pd.factorize(testdata[1])
testdata[2], test_services = pd.factorize(testdata[2])
testdata[3], test_flags = pd.factorize(testdata[3])
#testdata[41], test_attacks = pd.factorize(testdata[41])
testdata[41] = testdata[41].map(order)
test_attacks = np.array(list(order.keys()))

# Splitting the data into features and labels
X_train = traindata.iloc[:, :traindata.shape[1] - 1]  # Features of train set
y_train = traindata.iloc[:, traindata.shape[1] - 1:]  # Label of train set

X_train = pd.DataFrame(X_train)  # dataframe of features
y_train = y_train.values.ravel()  # array of labels

X_test = testdata.iloc[:, :testdata.shape[1] - 1]  # Features of test set
y_test = testdata.iloc[:, testdata.shape[1] - 1:]  # Labels of test set

X_test = pd.DataFrame(X_test)  # dataframe of features
y_test = y_test.values.ravel()  # array of labels

# count the number of normal and attack labels in the test set
normal_count = 0
attack_count = 0
for label in y_test:
    if label == 0:
        normal_count += 1
    else:
        attack_count += 1
print("Normal labels in test set:            ", normal_count)
print("Attack labels in test set (abnormal): ", attack_count)

# Print shapes
print("Shapes Info")
print("| Train Dataset shape:                  ", traindata.shape)
print("| Test Dataset shape:                   ", testdata.shape)
print("| Train Set FEATURES (X_train) shape:   ", X_train.shape)
print("| Test  Set FEATURES (X_test)  shape:   ", X_test.shape)
print("| Train Set LABEL (y_train) array size: ", y_train.size)
print("| Test  Set LABEL (y_test) array size : ", y_test.size)
print("\n")

print("Training model")

#clf = RandomForestClassifier(max_depth=10)
#clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
 #                            min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
  #                           min_impurity_decrease=0.0, class_weight=None)

clf = DecisionTreeClassifier()
# clf = Perceptron()

clf_name = type(clf).__name__
print("Using", clf_name)

trained_model = clf.fit(X_train, y_train)

training_score = trained_model.score(X_train, y_train)
print("Score (training set): ", training_score)

print("Predicting")
y_pred = clf.predict(X_test)

# print score of the model on the test set
testing_score = trained_model.score(X_test, y_test)
print("Score (test set):     ", testing_score)

# Score Table
h = ["TRAINING SET", "TEST SET"]
d = [[training_score, testing_score]]
t = tabulate(d, headers=h, tablefmt="fancy_grid", numalign="center")
print(t)

# Confusion Matrix (printing using tabulate)
results = confusion_matrix(y_test, y_pred)
# Definisci le etichette di classe
class_labels = ["Normal", "DOS", "R2L", "U2R", "Probing"]

# Aggiungi le etichette alla matrice di confusione
results_labeled = []
for i in range(len(results)):
    row_sum = np.sum(results[i])  # Somma della riga
    row_label = f"{class_labels[i]} ({row_sum})"
    row_percentages = [f'{value} ({value / row_sum:.2%})' if row_sum > 0 else f'{value}' for value in results[i]]
    row = [row_label] + row_percentages
    results_labeled.append(row)

# Stampa la matrice di confusione con etichette
print("Confusion matrix:")
print(tabulate(results_labeled, headers=[""] + class_labels, tablefmt="fancy_grid"))

