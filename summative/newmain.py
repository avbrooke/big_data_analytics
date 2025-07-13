# --- IMPORTS ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# --- DATA LOADING & CLEANING ---
def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df_clean = df.copy()

        # Fill missing values
        categorical_fill_none = ['Furniture', 'parking', 'amenities', 'appliances']
        for col in categorical_fill_none:
            df_clean[col] = df_clean[col].fillna('None')

        numerical_fill_median = ['Utility_payments', 'Pets_allowed', 'Children_are_welcome', 'Floor_area']
        for col in numerical_fill_median:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        categorical_fill_mode = ['Balcony', 'Renovation', 'Construction_type']
        for col in categorical_fill_mode:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        # Normalize currency to USD
        exchange_rates = {'AMD': 1, 'USD': 480, 'EUR': 520, 'RUB': 5.5}
        df_clean['Price_AMD'] = df_clean.apply(
            lambda row: row['Price'] * exchange_rates.get(row['Currency'], 1), axis=1)
        df_clean['Price_USD'] = df_clean['Price_AMD'] / 480

        # Clean string columns
        df_clean['Balcony'] = df_clean['Balcony'].str.lower().str.replace(' ', '_')
        df_clean['Renovation'] = df_clean['Renovation'].str.lower().str.replace(' ', '_')
        df_clean['Construction_type'] = df_clean['Construction_type'].str.lower().str.replace(' ', '_')

        # Fix Balcony categories
        balcony_mapping = {
            'open_balcony': 'open', 'open balcony': 'open',
            'closed_balcony': 'closed', 'closed balcony': 'closed',
            'multiple_balconies': 'multiple', 'multiple balconies': 'multiple',
            'not_available': 'none', 'not available': 'none', '0': 'none'
        }
        df_clean['Balcony'] = df_clean['Balcony'].replace(balcony_mapping)

        # Normalize price per day
        duration_mapping = {'daily': 1, 'weekly': 7, 'monthly': 30, 'yearly': 365}
        df_clean['Duration'] = df_clean['Duration'].str.lower()
        df_clean['Days'] = df_clean['Duration'].map(duration_mapping)
        df_clean = df_clean.dropna(subset=['Days'])
        df_clean['Price_per_day'] = df_clean['Price_USD'] / df_clean['Days']

        return df_clean

    except Exception as e:
        print(f"Data loading/cleaning error: {e}")
        return None

# --- CORRELATION ANALYSIS ---
def perform_correlation_analysis(df):
    df['Duration_numeric'] = df['Duration'].map({'daily': 1, 'weekly': 7, 'monthly': 30, 'yearly': 365})
    correlation_data = df[['Number_of_rooms', 'Price_per_day', 'Duration_numeric']]
    correlation_matrix = correlation_data.corr(method='spearman')

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Spearman Correlation Matrix')
    plt.tight_layout()
    plt.show()

    print("\nSpearman Correlation Coefficients (with p-values):")
    var_pairs = [('Number_of_rooms', 'Price_per_day'),
                 ('Number_of_rooms', 'Duration_numeric'),
                 ('Price_per_day', 'Duration_numeric')]

    for var1, var2 in var_pairs:
        rho, pval = spearmanr(df[var1], df[var2])
        print(f"{var1} ↔ {var2}: ρ = {rho:.4f}, p = {pval:.4e}")

# --- REGION EXTRACTION & GROUPING ---
def extract_and_group_regions(df):
    """
    Extracts broader regions from the last part of the address,
    transliterates to ASCII, and groups rare regions under 'other'.
    """
    # Use the LAST segment of the address — likely to be city or district
    df['Region'] = (
        df['Address'].astype(str)
        .str.split(',').str[-1]   # last component
        .str.strip()
        .apply(unidecode)         # transliterate (e.g., Cyrillic, Armenian)
        .str.lower()
    )

    # Group infrequent regions
    region_counts = df['Region'].value_counts()
    top_regions = region_counts[region_counts >= 30].index
    df['Region_grouped'] = df['Region'].apply(lambda x: x if x in top_regions else 'other')

    return df

# --- REGRESSION MODEL ---
def build_and_evaluate_model(df):
    features = ['Region_grouped', 'Renovation', 'Construction_type', 'Number_of_rooms']
    target = 'Price_per_day'
    X = df[features]
    y = df[target]

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

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nModel Performance (Grouped Regions):")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: ${mae:.2f} per day")

    # Boxplot
    filtered = df[df['Region_grouped'] != 'other']
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=filtered, x='Region_grouped', y='Price_per_day')
    plt.xticks(rotation=45)
    plt.title('Price per Day by Region (Grouped)')
    plt.ylabel('Price per Day (USD)')
    plt.tight_layout()
    plt.show()

# --- MAIN FUNCTION ---
def main():
    df_clean = load_and_clean_data('apartment_for_rent_train.csv')
    if df_clean is not None:
        # perform_correlation_analysis(df_clean)
        df_clean = extract_and_group_regions(df_clean)
        build_and_evaluate_model(df_clean)

# --- RUN ---
if __name__ == "__main__":
    main()