import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# load the data from the CSV file into a dataframe
weather_data = pd.read_csv('weather.nominal.csv', delimiter=',', encoding='utf-8')
# select the features used for the decision tree, and the column for the output
features = weather_data.iloc[:,:4]
output = weather_data.iloc[:,4]
# encode the categorical features into numeric values
features_encoded = pd.get_dummies(features)
# separate the dataset into training and testing set
features_train, features_test, output_train, output_test = train_test_split(features_encoded, output, random_state=0, test_size=0.4)
# adjusting parameters
decision_tree = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=0)
# build the decision tree by calling the fit method
decision_tree.fit(features_train, output_train)
# display the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plot_tree(decision_tree, feature_names=features_encoded.columns, class_names=decision_tree.classes_, filled=True)
import os

save_path = os.path.join(os.getcwd(), "decision_tree.png")
plt.savefig(save_path)
print(f"Saved to: {save_path}")