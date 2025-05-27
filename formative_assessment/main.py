import pandas as pd

df = pd.read_csv("cleaned_apartment_for_rent_train.csv")

# print(df.groupby('Pets_allowed')['Price_USD'].mean().sort_values(ascending=False))

# print(df['Elevator'].value_counts())

# print(df[['Number_of_rooms', 'Price_USD']].corr())

print(df.groupby('Duration')['Price_USD'].mean().sort_values(ascending=False))