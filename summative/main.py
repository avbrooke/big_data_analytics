"""
Apartment Rental Price Analysis
Script Performs data cleaning, statistical testing (Kruskal-Wallis, Spearman), and OLS regression on rental dataset.
Used in support of the Big Data Analytics report.
"""

# IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import kruskal
from unidecode import unidecode
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# DATA CLEANING
def clean_data(filepath):
    df = pd.read_csv(filepath)
    df_clean = df.copy()

    for col in ['Furniture', 'parking', 'amenities', 'appliances']:
        df_clean[col] = df_clean[col].fillna('None')
    for col in ['Utility_payments', 'Pets_allowed', 'Children_are_welcome', 'Floor_area']:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in ['Balcony', 'Renovation', 'Construction_type']:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    exchange_rates = {'AMD': 1, 'USD': 480, 'EUR': 520, 'RUB': 5.5}
    df_clean['Price_AMD'] = df_clean.apply(lambda row: row['Price'] * exchange_rates.get(row['Currency'], 1), axis=1)
    df_clean['Price_USD'] = df_clean['Price_AMD'] / 480

    for col in ['Balcony', 'Renovation', 'Construction_type']:
        df_clean[col] = df_clean[col].str.lower().str.replace(' ', '_')

    balcony_mapping = {
        'open_balcony': 'open', 'open balcony': 'open',
        'closed_balcony': 'closed', 'closed balcony': 'closed',
        'multiple_balconies': 'multiple', 'multiple balconies': 'multiple',
        'not_available': 'none', 'not available': 'none', '0': 'none'
    }
    df_clean['Balcony'] = df_clean['Balcony'].replace(balcony_mapping)

    duration_mapping = {'daily': 1, 'weekly': 7, 'monthly': 30, 'yearly': 365}
    df_clean['Duration'] = df_clean['Duration'].str.lower()
    df_clean['Days'] = df_clean['Duration'].map(duration_mapping)
    df_clean = df_clean.dropna(subset=['Days'])
    df_clean['Price_per_day'] = df_clean['Price_USD'] / df_clean['Days']
    df_clean['Duration_numeric'] = df_clean['Days']

    df_clean['Region'] = (
        df_clean['Address'].astype(str)
        .str.split(',').str[0]
        .str.strip()
        .apply(unidecode)
        .str.lower()
    )
    region_counts = df_clean['Region'].value_counts()
    top_regions = region_counts[region_counts >= 30].index
    df_clean['Region_grouped'] = df_clean['Region'].apply(lambda x: x if x in top_regions else 'other')

    return df_clean

# BOX PLOTS AND KRUSKAL WALLIS
def analyse_discrete_variable(df, column_name):
    print(f"\n--- {column_name} vs Price_per_day ---")

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[df['Price_per_day'] <= 150], x=column_name, y='Price_per_day')
    plt.xticks(rotation=45)
    plt.title(f'Price per Day by {column_name}')
    plt.ylabel('Price per Day (USD)')
    plt.tight_layout()
    plt.show()

    # Run Kruskal-Wallis test
    groups = [group['Price_per_day'].values for name, group in df.groupby(column_name)]
    h_stat, p_val = kruskal(*groups)
    print(f"Kruskal-Wallis H-test for {column_name}: H = {h_stat:.2f}, p = {p_val:.4e}")

# SPEARMAN CORRELATION
def run_spearman_correlation(df):
    print("\nSpearman Correlation Matrix:")
    corr_matrix = df[['Number_of_rooms', 'Price_per_day', 'Duration_numeric']].corr(method='spearman')
    print(corr_matrix)

    print("\nSpearman Correlation Coefficients (with p-values):")
    for var1, var2 in [('Number_of_rooms', 'Price_per_day'),
                       ('Number_of_rooms', 'Duration_numeric'),
                       ('Price_per_day', 'Duration_numeric')]:
        rho, pval = spearmanr(df[var1], df[var2])
        print(f"{var1} ↔ {var2}: ρ = {rho:.4f}, p = {pval:.4e}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Spearman Correlation Matrix')
    plt.tight_layout()
    plt.show()

# OLS REGRESSION
def run_ols_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    features = ['Region_grouped', 'Renovation', 'Construction_type', 'Number_of_rooms']
    target = 'Price_per_day'

    # Drop rows with missing target
    df = df.dropna(subset=[target])

    # One-hot encoding and cleaning
    df_encoded = pd.get_dummies(df[features], drop_first=True)
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaNs in predictors
    valid_idx = df_encoded.dropna().index
    X = df_encoded.loc[valid_idx]
    y = df.loc[valid_idx, target].astype(float)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add constant
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Convert to float for statsmodels compatibility
    X_train_const = X_train_const.astype(float)
    y_train = y_train.astype(float)

    # Fit the OLS model
    model = sm.OLS(y_train, X_train_const).fit()
    print(model.summary())

    # Evaluation on the test set
    y_pred = model.predict(X_test_const)
    print("\n--- Test Set Evaluation ---")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred):.2f} per day")

# MAIN FUNCTION
def main():
    df_clean = clean_data('apartment_for_rent_train.csv')

    # Run correlation and regression
    run_spearman_correlation(df_clean)
    run_ols_model(df_clean)

    # Run discrete variable analysis (box plots and H-test)
    for col in ['Balcony', 'Renovation', 'Construction_type']:
        analyse_discrete_variable(df_clean, col)

# RUN
if __name__ == "__main__":
    main()