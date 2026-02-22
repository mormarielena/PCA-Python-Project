import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_cleaned_data(filepath="input/global_economy_2021.csv"):
    # Load dataset
    df = pd.read_csv(filepath)

    # Define numerical variables
    numerical_columns = [
        'Population', 'Per_Capita_GNI', 'Agriculture_Forestry_Fishing',
        'Construction', 'Exports', 'Imports', 'Transport_Communication',
        'Retail_Trade_Hospitality', 'Gross_National_Income_USD', 'Gross_Domestic_Product'
    ]

    # Select numerical data
    numerical_data = df[numerical_columns]

    # Check for missing values and fill
    print("Missing values per column:")
    print(numerical_data.isnull().sum())
    numerical_data = numerical_data.fillna(numerical_data.mean())

    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_data)

    return numerical_data, scaled_data, numerical_columns