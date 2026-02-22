import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import bartlett


def run_factor_analysis(numerical_data, scaled_data, numerical_columns):
    # Bartlett's Test
    chi_square_value, p_value = bartlett(*[numerical_data[col] for col in numerical_columns])
    print("\nBartlett's Test of Sphericity:")
    print(f"Chi-square: {chi_square_value:.4f}, P-value: {p_value:.4f}")

    # KMO Index
    kmo_all, kmo_model = calculate_kmo(numerical_data)
    kmo_table = pd.DataFrame({'Variable': numerical_columns, 'KMO Index': kmo_all})
    print("\nKMO Index per variable:")
    print(kmo_table)
    kmo_table.to_csv("output/KMO_Indices.csv", index=False)

    # Factor Variance (No Rotation)
    fa = FactorAnalyzer(n_factors=2, rotation=None)
    fa.fit(scaled_data)
    ev, _ = fa.get_eigenvalues()
    explained_variance = ev[:2] / sum(ev) * 100
    variance_factors = pd.DataFrame({'Factor': ['Factor1', 'Factor2'], 'Explained Variance (%)': explained_variance})
    print("\nVariance explained by factors:")
    print(variance_factors)
    variance_factors.to_csv("output/Factor_Variance.csv", index=False)

    # Factor Analysis with Rotation (Varimax)
    fa_rotated = FactorAnalyzer(n_factors=2, rotation='varimax')
    fa_rotated.fit(scaled_data)

    # Loadings (Unrotated & Rotated)
    loadings_no_rot = pd.DataFrame(fa.loadings_, columns=['Factor1', 'Factor2'], index=numerical_columns)
    loadings_rot = pd.DataFrame(fa_rotated.loadings_, columns=['Factor1', 'Factor2'], index=numerical_columns)
    print("\nFactor Loadings (Rotated):")
    print(loadings_rot)
    loadings_no_rot.to_csv("output/Factor_Loadings_No_Rotation.csv")
    loadings_rot.to_csv("output/Factor_Loadings_Rotated.csv")

    # Factor Scores
    scores_no_rot = pd.DataFrame(fa.transform(scaled_data), columns=['Factor1', 'Factor2'])
    scores_rot = pd.DataFrame(fa_rotated.transform(scaled_data), columns=['Factor1', 'Factor2'])
    scores_no_rot.to_csv("output/Factor_Scores_No_Rotation.csv", index=False)
    scores_rot.to_csv("output/Factor_Scores_Rotated.csv", index=False)

    # Communalities
    fa_communalities = pd.DataFrame(fa_rotated.get_communalities(), index=numerical_columns, columns=['Communality'])
    print("\nFactor Analysis Communalities:")
    print(fa_communalities)
    fa_communalities.to_csv("output/Factor_Communalities.csv")

    # --- Visualizations ---

    # KMO Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kmo_table.set_index('Variable').T, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('KMO Indices Heatmap')
    plt.savefig("output/KMO_Heatmap.png")
    plt.show()

    # Rotated Loadings Plot
    plt.figure(figsize=(12, 8))
    for i in range(len(loadings_rot)):
        plt.arrow(0, 0, loadings_rot.iloc[i, 0], loadings_rot.iloc[i, 1], color='green', alpha=0.8)
        plt.text(loadings_rot.iloc[i, 0] * 1.1, loadings_rot.iloc[i, 1] * 1.1, loadings_rot.index[i], color='darkgreen')
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Factor Loadings (Rotated)')
    plt.savefig("output/Rotated_Factor_Loadings.png")
    plt.show()

    # Rotated Scores Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(scores_rot['Factor1'], scores_rot['Factor2'], alpha=0.7, color='orange')
    plt.title('Factor Scores (Rotated)')
    plt.grid(True)
    plt.savefig("output/Rotated_Factor_Scores.png")
    plt.show()