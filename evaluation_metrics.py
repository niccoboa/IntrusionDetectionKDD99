import numpy as np
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import data_info

import matplotlib.pyplot as plt
import seaborn as sns  # for heatmap visualization of confusion matrix

from sklearn.preprocessing import LabelEncoder


def calculate_pcc(y_test, y_pred):
    results = confusion_matrix(y_test, y_pred)
    correct_predictions = np.diag(results).sum()
    total_samples = results.sum()
    pcc = correct_predictions / total_samples
    return pcc


def print_confusion_matrix(y_test, y_pred):
    class_labels = ["Normal", "DOS", "R2L", "U2R", "Probing"]
    le = LabelEncoder()
    le.fit(class_labels)

    y_test_inverse = le.inverse_transform(y_test)  # Inverse transform to original class names (symbolic not numeric)
    y_pred_inverse = le.inverse_transform(y_pred)  # Inverse transform to original class names (symbolic not numeric)

    results = confusion_matrix(y_test_inverse, y_pred_inverse, labels=class_labels)

    # print(results)  # Print confusion matrix as a numpy array (uncomment to see)

    # Print confusion matrix in tabulate form
    results_labeled = []  # Confusion matrix with class labels
    for i in range(len(results)):
        row_sum = np.sum(results[i])  # Sum of row values (total samples for that class)
        row_label = f"{class_labels[i]} ({row_sum})"  # Class label and total samples of that class
        # Calculate row percentages and format them as strings
        row_percentages = [f'{value / row_sum:.2%} ({value})' if row_sum > 0 else f'{value}' for value in results[i]]
        row = [row_label] + row_percentages  # Concatenate row label and row percentages
        results_labeled.append(row)  # Append row to results_labeled

    print("Confusion matrix:")
    print(tabulate(results_labeled, headers=[""] + class_labels, tablefmt="fancy_grid"))

    # Create matplotlib heatmap of normalized confusion matrix
    cm_normalized = results.astype('float') / results.sum(axis=1)[:, np.newaxis]  # Normalize (each row sums to 1)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm_normalized, annot=True, cmap="Blues", fmt=".2%", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()


def print_score(training_score, testing_score):
    h = ["TRAINING SET", "TEST SET"]
    d = [[training_score, testing_score]]
    t = tabulate(d, headers=h, tablefmt="fancy_grid", numalign="center")
    print(t)


def print_pcc(pcc):
    print("Percent of Correct Classification (PCC): {:.2%}".format(pcc))  # Print PCC
