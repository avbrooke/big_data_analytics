# --- IMPORTS ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, spearmanr
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# --- LOAD DATA ---
train_df = pd.read_csv('apartment_for_rent_train.csv')
train_clean = train_df.copy()

# --- HANDLE MISSING VALUES ---
categorical_fill_none = ['Furniture', 'parking', 'amenities', 'appliances']
for col in categorical_fill_none:
    train_clean[col] = train_clean[col].fillna('None')

numerical_fill_median = ['Utility_payments', 'Pets_allowed', 'Children_are_welcome', 'Floor_area']
for col in numerical_fill_median:
    train_clean[col] = train_clean[col].fillna(train_clean[col].median())

categorical_fill_mode = ['Balcony', 'Renovation', 'Construction_type']
for col in categorical_fill_mode:
    train_clean[col] = train_clean[col].fillna(train_clean[col].mode()[0])

# --- NORMALIZE CURRENCY TO USD ---
exchange_rates = {'AMD': 1, 'USD': 480, 'EUR': 520, 'RUB': 5.5}
train_clean['Price_AMD'] = train_clean.apply(lambda row: row['Price'] * exchange_rates.get(row['Currency'], 1), axis=1)
train_clean['Price_USD'] = train_clean['Price_AMD'] / 480

# --- CLEAN STRINGS ---
train_clean['Balcony'] = train_clean['Balcony'].str.lower().str.replace(' ', '_')
train_clean['Renovation'] = train_clean['Renovation'].str.lower().str.replace(' ', '_')
train_clean['Construction_type'] = train_clean['Construction_type'].str.lower().str.replace(' ', '_')

# --- FIX BALCONY CATEGORY MAPPING ---
balcony_mapping = {
    'open_balcony': 'open', 'open balcony': 'open',
    'closed_balcony': 'closed', 'closed balcony': 'closed',
    'multiple_balconies': 'multiple', 'multiple balconies': 'multiple',
    'not_available': 'none', 'not available': 'none', '0': 'none'
}
train_clean['Balcony'] = train_clean['Balcony'].replace(balcony_mapping)

# --- NORMALIZE PRICE TO PER DAY ---
duration_mapping = {'daily': 1, 'weekly': 7, 'monthly': 30, 'yearly': 365}
train_clean['Duration'] = train_clean['Duration'].str.lower()
train_clean['Days'] = train_clean['Duration'].map(duration_mapping)
train_clean = train_clean.dropna(subset=['Days'])
train_clean['Price_per_day'] = train_clean['Price_USD'] / train_clean['Days']

# --- CORRELATION ANALYSIS ---
train_clean['Duration_numeric'] = train_clean['Duration'].map(duration_mapping)
correlation_data = train_clean[['Number_of_rooms', 'Price_per_day', 'Duration_numeric']]

correlation_matrix = correlation_data.corr(method='spearman')
print("\nSpearman Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Correlation Matrix')
plt.tight_layout()
# plt.show()

print("\nSpearman Correlation Coefficients (with p-values):")
var_pairs = [('Number_of_rooms', 'Price_per_day'),
             ('Number_of_rooms', 'Duration_numeric'),
             ('Price_per_day', 'Duration_numeric')]

for var1, var2 in var_pairs:
    rho, pval = spearmanr(train_clean[var1], train_clean[var2])
    print(f"{var1} ↔ {var2}: ρ = {rho:.4f}, p = {pval:.4e}")

# --- EXTRACT REGION FROM ADDRESS & GROUP ---
train_clean['Region'] = (
    train_clean['Address'].astype(str)
    .str.split(',').str[0]
    .str.strip()
    .apply(unidecode)
    .str.lower()
)

# --- GROUP RARE REGIONS ---
region_counts = train_clean['Region'].value_counts()
top_regions = region_counts[region_counts >= 30].index
train_clean['Region_grouped'] = train_clean['Region'].apply(lambda x: x if x in top_regions else 'other')

# --- REGRESSION MODEL WITH GROUPED REGION ---
features = ['Region_grouped', 'Renovation', 'Construction_type', 'Number_of_rooms']
target = 'Price_per_day'
X = train_clean[features]
y = train_clean[target]

categorical_features = ['Region_grouped', 'Renovation', 'Construction_type']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- EVALUATE MODEL ---
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel Performance (Grouped Regions):")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: ${mae:.2f} per day")

# --- VISUAL CHECK: BOX PLOT BY GROUPED REGION ---
filtered = train_clean[train_clean['Region_grouped'] != 'other']
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered, x='Region_grouped', y='Price_per_day')
plt.xticks(rotation=45)
plt.title('Price per Day by Region (Grouped)')
plt.ylabel('Price per Day (USD)')
plt.tight_layout()
plt.show()
