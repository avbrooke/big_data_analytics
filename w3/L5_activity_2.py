from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

weather_data = pd.read_csv('weather.nominal.csv', delimiter=',',
encoding='utf-8')
features = weather_data.iloc[:,:4]
output = weather_data.iloc[:,4]
# One-hot encode features
features_encoded = pd.get_dummies(features)

# create decision tree
decision_tree = DecisionTreeClassifier(criterion="entropy",
splitter="best", random_state=0)
decision_tree.fit(features_encoded, output)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plot_tree(decision_tree, feature_names=features_encoded.columns,
class_names=decision_tree.classes_, filled=True)
# plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_predict = decision_tree.predict(features_encoded)
confusion_m = confusion_matrix(output,y_predict,
labels=decision_tree.classes_)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_m,
display_labels=['No', 'Yes'])
plot1 = cm_display.plot()

features_train, features_test, output_train, output_test = train_test_split(features_encoded, output, random_state=3, test_size=0.3)
decision_tree.fit(features_train, output_train)
y_predict = decision_tree.predict(features_test)
confusion_m = confusion_matrix(output_test,y_predict,
labels=decision_tree.classes_)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_m,
display_labels=['No', 'Yes'])
plot = cm_display.plot()
# plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, cohen_kappa_score, roc_curve
# Confusion Matrix
tn, fp, fn, tp = confusion_m.ravel()
# probability for positive class
y_prob = decision_tree.predict_proba(features_test)[:, 1]
# Precision
precision = precision_score(output_test, y_predict, pos_label='yes')
# # Recall (a.k.a. True Positive Rate)
recall = recall_score(output_test, y_predict, pos_label='yes')
# # F1 Score
f1 = f1_score(output_test, y_predict, pos_label='yes')
# False Positive Rate (FPR)
fpr = fp / (fp + tn)
# # ROC AUC Score
roc_auc = roc_auc_score(output_test, y_prob)
# Kappa Statistics
kappa = cohen_kappa_score(output_test, y_predict)
# Output metrics
print(f"Precision: {precision:.2f}")
print(f"Recall (TPR): {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"False Positive Rate (FPR): {fpr:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")
print(f"Cohen's Kappa: {kappa:.2f}")

# cross validation
from sklearn.model_selection import cross_val_score
import numpy as np
# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
# Perform 5-fold cross-validation
cv_scores = cross_val_score(clf, features_encoded, output, cv=4)
# Print results
print("Cross-validation scores:", cv_scores)

# more detailed cross validation:
from sklearn.model_selection import StratifiedKFold
# Cross-validation setup
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
# Metric storage
precisions, recalls, f1s, aucs, kappas, fprs = [], [], [], [], [], []
fold_number = 0
# Iterating through each fold to build the model and measure it performance
# A common convention used in sklearn documentation is to use X for the features
# and y for the output. In previous code I used "features" and "output" to facilitate
# understanding, however you can use the sklearn convention in your own work.
for train_idx, test_idx in cv.split(features_encoded, output):
# _idx represents the indices of the rows selected for each subsets.
    X_train, X_test = features_encoded.iloc[train_idx], features_encoded.iloc[test_idx]
    y_train, y_test = output.iloc[train_idx], output.iloc[test_idx]
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
# Compute metrics
    precision = precision_score(y_test, y_pred, pos_label='yes',
    zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label='yes', zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label='yes', zero_division=0)
    single_class_in_fold = False
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
    # Case where only one class is present in the set for that fold.
    # In that case the roc_auc cannot be computed.
        single_class_in_fold = True

    kappa = cohen_kappa_score(y_test, y_pred)
    # Build the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)
    # Create a DataFrame with labels
    labels = ['No', 'Yes']
    confusion_matrix_table = pd.DataFrame(confusion_mat, index=[f'True{label}' for label in labels],
    columns=[f'Predicted {label}' for label in
    labels])
    # False Positive Rate
    tn, fp, fn, tp = confusion_mat.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    # Append metrics. This will be used outside of the for loop to compute the average
    # performance over all folds.
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    aucs.append(roc_auc)
    kappas.append(kappa)
    fprs.append(fpr)
    print('-------------------------------------')
    print(f"fold {fold_number} performance:")
    print("Confusion matrix:\n")
    print(confusion_matrix_table)
    print(f"\n\nPrecision: {precision:.2f}")
    print(f"Recall (TPR): {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"False Positive Rate (FPR): {fpr:.2f}")
    if not single_class_in_fold:
        print(f"ROC AUC Score: {roc_auc:.2f}")
    print(f"Cohen's Kappa: {kappa:.2f}")
    print('-------------------------------------')
    fold_number += 1

# Display average results
print(f"Average performance across all {fold_number} folds:")
print(f"Precision: {np.mean(precisions):.2f}")
print(f"Recall (TPR): {np.mean(recalls):.2f}")
print(f"F1 Score: {np.mean(f1s):.2f}")
print(f"False Positive Rate (FPR): {np.mean(fprs):.2f}")
print(f"ROC AUC Score: {np.mean(aucs):.2f}")
print(f"Cohen's Kappa: {np.mean(kappas):.2f}")

#                             Evaluation:

#                 Percentage Split  | Cross-Validation
#                     0.5 | 0.6 | 0.7 | Fold0| Fold1| Fold2| Fold3
#        Precision |  0.50| 0.33| 0.50| 1.00 | 0.67 | 0.50 | 1.00
#     Recall (TPR) |  1.00| 0.33| 1.00| 0.50 | 0.67 | 0.50 | 0.50
#         F1 Score |  0.67| 0.33| 0.67| 0.67 | 0.67 | 0.50 | 0.67
#              FPR |  0.75| 0.67| 0.67| 0.00 | 1.00 | 1.00 | 0.00
#    ROC AUC score |  0.62| 0.33| 0.67| 0.75 | 0.33 | 0.25 | 0.75
#    Cohen's Kappa |  0.22|-0.33| 0.29| 0.50 |-0.33 |-0.50 | 0.40

# Best overall cross-validation performance: Fold 0 and Fold 3 show strong metrics with perfect precision and no false positives.
# Worst performance: Fold 2 and Fold 1- low Kappa and high FPR. This shows that even with cross-validation, fold splits can vary widely in small datasets.
# 60% split under performs: Especially low on all metrics likely due to poor sampling or unlucky distribution (this can happen in small datasets).
# Kappa Scores: These show how much better your model is than random guessing. Values near or below 0 suggest random-like performance- not ideal!


