import pandas as pd

df_train = pd.read_csv("apartment_for_rent_train.csv")
df_test = pd.read_csv("apartment_for_rent_test.csv")

print(df_train.head())