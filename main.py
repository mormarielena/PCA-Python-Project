import os
from utils import get_cleaned_data
from pca_analysis import run_pca_analysis
from factor_analysis import run_factor_analysis

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs("output")

    # 1. Loading data
    num_data, scaled_data, columns = get_cleaned_data("input/global_economy_2021.csv")

    # 2. PCA Analysis
    print("\n1 -- Running PCA...")
    run_pca_analysis(num_data, scaled_data, columns)

    # 3. Factor Analysis
    print("\n2 -- Running Factor Analysis...")
    run_factor_analysis(num_data, scaled_data, columns)

    print("\nAnalysis complete. Check the 'output/' folder for all results.")

if __name__ == "__main__":
    main()