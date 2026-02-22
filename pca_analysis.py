import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def run_pca_analysis(numerical_data, scaled_data, numerical_columns):
    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)

    # 1. Explained Variance Distribution
    explained_variance_ratio = pd.DataFrame({
        'Component': [f'PC{i + 1}' for i in range(len(pca.explained_variance_ratio_))],
        'Explained Variance (%)': pca.explained_variance_ratio_ * 100
    })
    print("\nExplained Variance Distribution:")
    print(explained_variance_ratio)
    explained_variance_ratio.to_csv("output/Variance_Distribution.csv", index=False)

    # 2. Principal Component Scores
    scores = pd.DataFrame(pca_result, columns=[f'PC{i + 1}' for i in range(len(pca.components_))])
    print("\nPrincipal Component Scores (Head):")
    print(scores.head())
    scores.to_csv("output/Scores.csv", index=False)

    # 3. Correlations (Loadings)
    loadings = pd.DataFrame(pca.components_.T,
                            columns=[f'PC{i + 1}' for i in range(len(pca.components_))],
                            index=numerical_columns)
    print("\nCorrelations between variables and components:")
    print(loadings)
    loadings.to_csv("output/Correlations.csv")

    # 4. Variable Contributions
    contributions = loadings ** 2
    print("\nVariable Contributions to each component:")
    print(contributions)
    contributions.to_csv("output/Contributions.csv")

    # 5. Squared Cosines
    squared_cosines = contributions.div(contributions.sum(axis=1), axis=0)
    print("\nSquared Cosines:")
    print(squared_cosines)
    squared_cosines.to_csv("output/Squared_Cosines.csv")

    # 6. Communalities
    communalities = contributions.sum(axis=1)
    print("\nCommunalities:")
    print(communalities)
    communalities.to_csv("output/Communalities.csv")

    # --- PCA Visualizations ---

    # Detailed Scree Plot
    explained_variance_ratio['Cumulative Explained Variance (%)'] = explained_variance_ratio[
        'Explained Variance (%)'].cumsum()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=explained_variance_ratio['Component'], y=explained_variance_ratio['Explained Variance (%)'],
                color='skyblue', label='Individual Variance')
    plt.plot(explained_variance_ratio['Component'], explained_variance_ratio['Cumulative Explained Variance (%)'],
             marker='o', color='darkblue', label='Cumulative Variance')
    plt.axhline(y=1 / len(numerical_columns) * 100, color='red', linestyle='--', label='Kaiser Criterion')
    plt.axhline(y=5, color='green', linestyle='--', label='Min Variance (5%)')
    plt.axvline(x=2.5, color='orange', linestyle='--', label='Cattell Criterion')
    plt.title('Explained Variance Distribution and Selection Criteria')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/Detailed_Scree_Plot.png")
    plt.show()

    # Scores Plot (PC1 vs PC2)
    plt.figure(figsize=(10, 6))
    plt.scatter(scores['PC1'], scores['PC2'], alpha=0.7, color='purple')
    plt.title('Principal Component Scores Plot (PC1 vs PC2)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("output/Scores_Plot.png")
    plt.show()

    # Correlations/Loadings Plot
    plt.figure(figsize=(12, 8))
    for i in range(len(loadings)):
        plt.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1], color='blue', alpha=0.8)
        plt.text(loadings.iloc[i, 0] * 1.1, loadings.iloc[i, 1] * 1.1, loadings.index[i], color='darkblue', fontsize=12)
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title('Variable Correlations with Principal Components (PC1 vs PC2)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("output/Correlations_Plot.png")
    plt.show()

    # Communalities Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(communalities.to_frame(name='Communality').T, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Communalities Heatmap')
    plt.tight_layout()
    plt.savefig("output/Communalities_Heatmap.png")
    plt.show()