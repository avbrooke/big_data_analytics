import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# load data
possum_data = pd.read_csv("possum.csv", delimiter=",", encoding= "utf-8")
independent_variable = possum_data[['totalL']]
dependant_variable = possum_data['headL']

# create the model
model = LinearRegression()

# split data
for split in [0.8, 0.7, 0.6, 0.5]:
    X_train, X_test, y_train, y_test = train_test_split(independent_variable, dependant_variable, test_size=split, random_state=42)

    # fit model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('-----------------------------------------------------------------')
    print(f'For a split of {round((1-split)*100)}% training',
    f'/ {round(split*100)}% test the model is:',
    f' y = {round(model.intercept_,2)} + {round(model.coef_[0],2)} x.')
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"R² Score: {r2:.3f}")
    print('-----------------------------------------------------------------')

# cross validation
from sklearn.model_selection import cross_validate

scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
folds = 3
results = cross_validate(model, independent_variable, dependant_variable, cv=folds, scoring=scoring, return_estimator=True)

for i in range(folds):
    print('-----------------------------------------------------------------')
    print(f"Fold {i+1} y = {results['estimator'][i].coef_} x +",
    f"{results['estimator'][i].intercept_}")
    print("MSE per fold:", -results['test_neg_mean_squared_error'][i])
    print("MAE per fold:", -results['test_neg_mean_absolute_error'][i])
    print("R² per fold:", results['test_r2'][i])
    print('-----------------------------------------------------------------')

    #            Percentage Split  | Cross-Validation
    #            0.5| 0.6| 0.7| 0.8| Fold1| Fold2| Fold3
    #     MSE | 4.64|4.47|6.53|9.23| 4.19 | 10.37| 9.88
    #     MAE | 1.62|1.59|1.88|2.33| 1.62 | 2.47 |2.51
    #      R2 | 0.45|0.53|0.46|0.27| -0.75| 0.38 |0.22
