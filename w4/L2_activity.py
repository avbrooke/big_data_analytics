import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# implement additional functions
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def root_relative_squared_error(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.sqrt(numerator / denominator)

def relative_absolute_error(y_true, y_pred):
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true - np.mean(y_true)))
    return numerator / denominator

# prepare the data
# load data
possum_data=pd.read_csv('../w3/possum.csv', delimiter=',', encoding='utf-8')
possum_encoded = pd.get_dummies(possum_data)
# the list of features used by the model
features_data = possum_encoded[['site', 'age', 'headL', 'skullW ', 'totalL',
'sex_m', 'sex_f']]
# the target (output) of the analysis
target_data = possum_encoded['tailL']

# create SVR and linear regression models
X_train, X_test, y_train, y_test = features_data, features_data, target_data, target_data

# create and train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# standardising features (important for SVR)
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)

# standardising target (important for SVR)
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(
y_train.values.reshape(len(y_train),1))

# create and train the SMOreg equivalent model
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_scaled, np.reshape(y_train_scaled, len(y_train_scaled)))

# predict for Linear Regression Model
lr_pred = lr_model.predict(X_test)

# predict for SVR Model
svr_pred = svr_model.predict(feature_scaler.transform(X_test))

# reverse scaling to evaluate model
svr_pred = target_scaler.inverse_transform(svr_pred.reshape(-1, 1)).flatten()

# evaluate
print('---------------- LR Model -----------------------------')
print("RRSE:", root_relative_squared_error(y_test, lr_pred))
print("RAE:", relative_absolute_error(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("R² score:", r2_score(y_test, lr_pred))
print('MSA:', mean_absolute_error(y_test, lr_pred))
print('---------------- SVR Model -----------------------------')
print("RRSE:", root_relative_squared_error(y_test, svr_pred))
print("RAE:", relative_absolute_error(y_test, svr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, svr_pred)))
print("R² score:", r2_score(y_test, svr_pred))
print('MSA:', mean_absolute_error(y_test, svr_pred))

# evaluating using percentage split
split_sizes = [0.2, 0.3, 0.4, 0.5]
for size in split_sizes:
    print(f'######### Percentage Split: {round(size*100)}% #########')
    # splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_data, target_data,
    test_size=size, random_state=19)
    # Create and train the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    # Predict for Linear Regression Model
    lr_pred = lr_model.predict(X_test)
    print('---------------- LR Model -----------------------------')
    print("RRSE:", root_relative_squared_error(y_test, lr_pred))
    print("RAE:", relative_absolute_error(y_test, lr_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))
    print("R² score:", r2_score(y_test, lr_pred))
    print('MSA:', mean_absolute_error(y_test, lr_pred))
    # Standardising features for SVR
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    # Standardising target SVR
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(
    y_train.values.reshape(len(y_train),1))
    # Create and train the SMOreg equivalent model
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(X_train_scaled, np.reshape(y_train_scaled, len(y_train_scaled)))
    # Predict for SVR Model
    svr_pred = svr_model.predict(feature_scaler.transform(X_test))
    # Inverse the transform on the predicted value so it can be compared to the
    # real data.
    svr_pred = target_scaler.inverse_transform(svr_pred.reshape(-1, 1)).flatten()
    print('---------------- SVR Model -----------------------------')
    print("RRSE:", root_relative_squared_error(y_test, svr_pred))
    print("RAE:", relative_absolute_error(y_test, svr_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, svr_pred)))
    print("R² score:", r2_score(y_test, svr_pred))
    print('MSA:', mean_absolute_error(y_test, svr_pred))
