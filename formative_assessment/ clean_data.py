import pandas as pd

# Load raw data
df_train = pd.read_csv("apartment_for_rent_train.csv")

# Start with a copy to work on
df = df_train.copy()

# Fill essential numeric fields
for col in ['Floor_area', 'Number_of_rooms', 'Number_of_bathrooms', 'Ceiling_height', 'Floor']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Fill boolean flags (assume missing = False/0)
for col in ['New_construction', 'Elevator', 'Children_are_welcome', 'Pets_allowed', 'Utility_payments']:
    df[col] = df[col].fillna(0).astype(int).astype(bool)

# Clean categorical text
for col in ['Construction_type', 'Balcony', 'Furniture', 'Renovation']:
    df[col] = df[col].fillna('unknown').str.strip().str.lower()

# Fix typos in renovation types
df['Renovation'] = df['Renovation'].replace({
    'euro_renovation': 'euro renovation',
    'cosmetic_renovation': 'cosmetic renovation',
    'major renovation': 'major renovation'
})

# Drop rows with missing price
df = df.dropna(subset=['Price'])

# Fill missing Currency and Duration
df['Currency'] = df['Currency'].fillna('AMD')
df['Duration'] = df['Duration'].fillna('monthly')

conversion_rates = {'AMD': 0.0025, 'USD': 1}
df['Price_USD'] = df.apply(
    lambda row: row['Price'] * conversion_rates.get(row['Currency'], 1),
    axis=1
)

# Convert Datetime
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce', dayfirst=True)

# Address fallback and region extraction
df['Address'] = df['Address'].fillna('unknown')
df['Region'] = df['Address'].str.extract(r'(›\s*[\w\s]+)', expand=False)
df['Region'] = df['Region'].fillna('Yerevan').str.strip().str.lower()

df.to_csv("cleaned_apartment_for_rent_train.csv", index=False)