from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load the dataset
data = load_iris()
# the data containing the features used the classification is in the field 'data'
# the training and testing sets are the same as we use the full data set for training
X_train = X_test = data.data
# the data containing the classified instances is in the field target
# again, the training and testing sets are the same as we use the full data set for training
y_train = y_test= data.target

# train a Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
# make predictions and evaluate
y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report for Full training set:\n",
      classification_report(y_test, y_pred, target_names=data.target_names))

# naive bayes model
modelNB = GaussianNB()
modelNB.fit(X_train, y_train)
# predict on the test set
y_pred = modelNB.predict(X_test)
# evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report for Naive Bayes Model:\n",
classification_report(y_test, y_pred, target_names=data.target_names))

#  SVM model
# feature scaling (recommended for SVM)
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
# create and train an SVM classifier using SMO (implicitly)
model = SVC(kernel='linear') # or 'rbf', 'poly' depending on your preference
model.fit(X_train_scaled, y_train)
# predict on the test set, which needs to be scaled using the transform method
y_pred = model.predict(feature_scaler.transform(X_test))
# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n",
      classification_report(y_test, y_pred, target_names=data.target_names))

# Split into training and test sets
for test_size in [0.2, 0.3, 0.4, 0.5]:
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=test_size, random_state=0)
    # train a Decision Tree
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    # make predictions and evaluate
    y_pred = clf.predict(X_test)
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    print(f'\n\n################################# Report for test size of {round(test_size*100)}% ###############################\n')
    print("\nClassification Report Decision Tree Model:\n",
          classification_report(y_test, y_pred, target_names=data.target_names))

    modelNB = GaussianNB()
    modelNB.fit(X_train, y_train)
    # predict on the test set
    y_pred = modelNB.predict(X_test)
    # evaluation
    print(' --------------------------------------- ')
    print("Classification Report for Naive Bayes Model:\n",
    classification_report(y_test, y_pred, target_names=data.target_names))
    # feature scaling (recommended for SVM)
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    # create and train an SVM classifier using SMO (implicitly)
    model = SVC(kernel='linear') # or 'rbf', 'poly' depending on your preference
    model.fit(X_train_scaled, y_train)
    # predict on the test set, which needs to be scaled using the transform method
    y_pred = model.predict(feature_scaler.transform(X_test))
    # evaluation
    print(' ------------------------------------- ')
    print("Classification Report for SVM Model:\n",
          classification_report(y_test, y_pred, target_names=data.target_names))