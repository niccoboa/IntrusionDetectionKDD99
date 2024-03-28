import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data10', header=None)


print(df.shape)

df.drop_duplicates(keep='first', inplace=True)
print(df.shape)

# The CSV file has no column heads, so add them
df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]

print(df.columns)
# Analyze the Dataset

print('='*42)
print("Dataset has {} rows.".format(len(df)))
print('The number of features are:', df.shape[1])
df['outcome'] = df['outcome'].str.rstrip('.')
output = df['outcome'].values
labels = set(output)
print('Attacks labels:', labels)
print('There are {} different labels'.format(len(labels)))


# Data Cleaning: checking for NULL values
null_rows_count = len(df[df.isnull().any(axis=1)])
print('Null values in dataset are', null_rows_count)

# Data analysis
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

# class distribution pl
#class_distribution_percent = df['outcome'].value_counts(normalize=True) * 100
#plt.figure(figsize=(15,7))
#class_distribution_percent.plot(kind='bar')
#plt.xlabel('Class')
#plt.ylabel('Data points per Class')
#plt.title('Distribution of yi in train data')
#plt.grid()
#plt.show()

# number of data points in each class and their percentages
#df['outcome'].value_counts()
#print(df['outcome'].value_counts(normalize=True) * 100)



input_cols = list(df.columns)[1:-1]
target_col = 'outcome'
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:-1]

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
scaler = MinMaxScaler()
scaler.fit(df[numeric_cols])
df[numeric_cols] = scaler.transform(df[numeric_cols])

le = LabelEncoder()

target = df['outcome']
df['outcome'] = le.fit_transform(target)
df['protocol_type'] = le.fit_transform(df['protocol_type'])
df['service'] = le.fit_transform(df['service'])
df['flag'] = le.fit_transform(df['flag'])






from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(train_df.shape)
print(test_df.shape)

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()



from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(n_estimators=5, random_state=42))
sel.fit(train_inputs, train_targets)
selected_feat = train_inputs.columns[(sel.get_support())]
print(selected_feat)
print(len(selected_feat))


# pred
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(train_inputs[selected_feat], train_targets);
preds_rf = rf.predict(test_inputs[selected_feat])
score_rf = accuracy_score(test_targets, preds_rf)
print("Accuracy of Random Forests: ", score_rf)

from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier()
dc.fit(train_inputs[selected_feat], train_targets);
preds_dc = dc.predict(test_inputs[selected_feat])
score_dc = accuracy_score(test_targets, preds_dc)
print("Accuracy of Decision Tree: ", score_dc)

