# --- IMPORTS ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

# --- LOAD DATA ---
train_df = pd.read_csv('apartment_for_rent_train.csv')

# --- COPY DATA FOR CLEANING ---
train_clean = train_df.copy()

# --- HANDLE MISSING VALUES ---
# Fill categorical columns with 'None'
categorical_fill_none = ['Furniture', 'parking', 'amenities', 'appliances']
for col in categorical_fill_none:
    train_clean[col] = train_clean[col].fillna('None')

# Fill numerical columns with median
numerical_fill_median = ['Utility_payments', 'Pets_allowed', 'Children_are_welcome', 'Floor_area']
for col in numerical_fill_median:
    median_value = train_clean[col].median()
    train_clean[col] = train_clean[col].fillna(median_value)

# Fill specific categorical columns with mode
categorical_fill_mode = ['Balcony', 'Renovation', 'Construction_type']
for col in categorical_fill_mode:
    mode_value = train_clean[col].mode()[0]
    train_clean[col] = train_clean[col].fillna(mode_value)

# --- NORMALIZE CURRENCIES TO AMD ---
exchange_rates = {
    'AMD': 1,
    'USD': 480,    # Example rates
    'EUR': 520,
    'RUB': 5.5
}

def convert_to_amd(row):
    rate = exchange_rates.get(row['Currency'], 1)  # Default rate is 1 if unknown
    return row['Price'] * rate

train_clean['Price_AMD'] = train_clean.apply(convert_to_amd, axis=1)

# --- CONVERT AMD TO USD ---
amd_to_usd_rate = 1 / 480  # 1 AMD = 0.002083 USD
train_clean['Price_USD'] = train_clean['Price_AMD'] * amd_to_usd_rate

# --- CLEAN BALCONY VALUES ---
train_clean['Balcony'] = train_clean['Balcony'].str.lower().str.replace(' ', '_')

balcony_mapping = {
    'open_balcony': 'open',
    'open balcony': 'open',
    'closed_balcony': 'closed',
    'closed balcony': 'closed',
    'multiple_balconies': 'multiple',
    'multiple balconies': 'multiple',
    'not_available': 'none',
    'not available': 'none',
    '0': 'none'
}
train_clean['Balcony'] = train_clean['Balcony'].replace(balcony_mapping)

train_clean['Construction_type'] = train_clean['Construction_type'].str.lower()

train_clean['Renovation'] = train_clean['Renovation'].str.lower().str.replace(' ', '_')

# --- DROP NaN Price Entries ---
train_clean = train_clean.dropna(subset=['Price_USD'])

# ANALYSIS: Average Price by Renovation
renovation_price_usd = train_clean.groupby('Renovation')['Price_USD'].mean()
print("\nAverage Price (USD) by Renovation Type:")
print(renovation_price_usd)

# PLOT: Boxplot of Price by Renovation (in USD)
filtered_data = train_clean[train_clean['Price_USD'] < 2500]  # Filter for visualization

plt.figure(figsize=(10, 5))
sns.boxplot(data=filtered_data, x='Renovation', y='Price_USD')
plt.title('Price Distribution by Renovation Type (USD)')
plt.xticks(rotation=45)
plt.ylabel('Price (USD)')
# plt.show()

# STATISTICAL TEST: Kruskal-Wallis H-Test for Renovation
categories = train_clean['Renovation'].unique()
groups = [train_clean[train_clean['Renovation'] == category]['Price_USD'] for category in categories]

kruskal_result = kruskal(*groups)
print(f"\nKruskal-Wallis test for Renovation vs Price_USD: H = {kruskal_result.statistic:.4f}, p = {kruskal_result.pvalue:.4e}")

if kruskal_result.pvalue < 0.05:
    print("Significant difference between Renovation type and Price (reject H0).")
else:
    print("No significant difference between Renovation type and Price (fail to reject H0).")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# --- Step 1: Map Duration to Numeric Values ---
# Map rental durations to approximate days
duration_mapping = {
    'daily': 1,
    'weekly': 7,
    'monthly': 30,
    'yearly': 365
}

# First, lower the duration column and map
train_clean['Duration_numeric'] = train_clean['Duration'].str.lower().map(duration_mapping)

# Check if any are still NaN (for unexpected categories)
print(train_clean['Duration_numeric'].isnull().sum())

# Optional: Drop rows where Duration couldn't be mapped
train_clean = train_clean.dropna(subset=['Duration_numeric'])

# --- Step 2: Prepare Data for Correlation ---
correlation_data = train_clean[['Number_of_rooms', 'Price_USD', 'Duration_numeric']]

# --- Step 3: Spearman Correlation Matrix ---
correlation_matrix = correlation_data.corr(method='spearman')
print("\nSpearman Correlation Matrix:")
print(correlation_matrix)

# --- Step 4: Visualize Correlation Matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Correlation Matrix: Number_of_rooms, Price, Duration')
plt.show()

# --- Step 5: Additional - Spearman's Rho and p-values (optional, more detailed) ---
# Example for two specific variables
rho, pval = spearmanr(train_clean['Number_of_rooms'], train_clean['Price_USD'])
print(f"\nSpearman correlation between Number_of_rooms and Price_USD: rho = {rho:.4f}, p = {pval:.4e}")

rho, pval = spearmanr(train_clean['Number_of_rooms'], train_clean['Duration_numeric'])
print(f"Spearman correlation between Number_of_rooms and Duration: rho = {rho:.4f}, p = {pval:.4e}")

rho, pval = spearmanr(train_clean['Price_USD'], train_clean['Duration_numeric'])
print(f"Spearman correlation between Price_USD and Duration: rho = {rho:.4f}, p = {pval:.4e}")