import pandas as pd
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

# Must declare data_dir as the directory of training and test files
data_dir = "./"
raw_data_filename = data_dir + "data10"

print("Loading raw data")

raw_data = pd.read_csv(raw_data_filename, header=None)

print("Transforming data")
# Categorize columns: "protocol", "service", "flag", "attack_type"
# Note: is_host_login and is_guest_login are symbolic as well, but they take only two values (like binary)
raw_data[1], protocols = pd.factorize(raw_data[1])
raw_data[2], services = pd.factorize(raw_data[2])
raw_data[3], flags = pd.factorize(raw_data[3])

# gather all attack types in 5 different class depending on the type of attack. The classes are: normal, dos, probe, r2l, u2r
# normal: normal
# dos: back, land, neptune, pod, smurf, teardrop, apache2, udpstorm, processtable, worm
# probe: ipsweep, nmap, portsweep, satan, mscan, saint
# r2l: guess_passwd, ftp_write, imap, phf, multihop, warezmaster, warezclient, spy, xlock, xsnoop, snmpguess, snmpgetattack, httptunnel, sendmail, named
# u2r: buffer_overflow, loadmodule, rootkit, perl, sqlattack, xterm, ps, httptunnel

raw_data[41] = raw_data[41].str.rstrip('.')
raw_data[41] = raw_data[41].replace(
    {
        'back': 'dos',
        'buffer_overflow': 'u2r',
        'ftp_write': 'r2l',
        'guess_passwd': 'r2l',
        'imap': 'r2l',
        'ipsweep': 'probe',
        'land': 'dos',
        'loadmodule': 'u2r',
        'multihop': 'r2l',
        'neptune': 'dos',
        'nmap': 'probe',
        'perl': 'u2r',
        'phf': 'r2l',
        'pod': 'dos',
        'portsweep': 'probe',
        'rootkit': 'u2r',
        'satan': 'probe',
        'smurf': 'dos',
        'spy': 'r2l',
        'teardrop': 'dos',
        'warezclient': 'r2l',
        'warezmaster': 'r2l'
    }
)

import numpy as np

unique_values = np.unique(raw_data[41])
print("Possible attacks:", unique_values)

raw_data[41], attacks = pd.factorize(raw_data[41])

# separate features (columns 1..40) and label (last column, the 41st)
X = raw_data.iloc[:, :raw_data.shape[1] - 1]  # Features
y = raw_data.iloc[:, raw_data.shape[1] - 1:]  # Label

# convert them into numpy arrays or matrices (dataframes)
df = pd.DataFrame(X)  # dataframe of features
y = y.values.ravel()  # array of labels

# TODO: get features names and target name

# Separate data in train set and test set
# Note: train_size + test_size < 1.0 means we are subsampling
# Use small numbers for slow classifiers, as KNN, Radius, SVC,...
X_train, X_test, y_train, y_test = train_test_split(df, y, train_size=0.9, test_size=0.1)
print("X_train, y_train size:", X_train.shape, "(", y_train.size, ")")
print("X_test, y_test:", X_test.shape, y_test.size)

# Training, choose model by commenting/uncommenting clf=
print("Training model")
# clf= RandomForestClassifier(n_jobs=-1, random_state=3, n_estimators=102)#, max_features=0.8, min_samples_leaf=3, n_estimators=500, min_samples_split=3, random_state=10, verbose=1)
clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                             min_impurity_decrease=0.0, class_weight=None)

trained_model = clf.fit(X_train, y_train)

# print the score of the model on the training set
print("Score (training set): ", trained_model.score(X_train, y_train))

# Predicting
print("Predicting")
y_pred = clf.predict(X_test)

print("Computing performance metrics")
results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)

print("Confusion matrix:\n", results)
print("Error: ", error)

# Calculate precision
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred, average='micro')
print('precision_score: ', precision)
