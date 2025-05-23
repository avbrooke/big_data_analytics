import pandas as pd
from sklearn.linear_model import LinearRegression
# load the data from the CSV file
possum_data=pd.read_csv('possum.csv', delimiter=',', encoding='utf-8')

# build our model
independent_variable = possum_data[['totalL']]
dependant_variable = possum_data['headL']

# create and fit the model
model = LinearRegression()
model.fit(independent_variable, dependant_variable)

# print model parameter (rounded to two decimal)
print(f'Intercept: {round(model.intercept_,2)}') # Intercept: 43.26
print(f'Coefficient: {round(model.coef_[0],2)}') # Coefficient: 0.57

test=pd.DataFrame({'totalL':[70, 80, 85]})
prediction = model.predict(test)
# print(prediction)