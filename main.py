import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import bartlett
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo

#Citirea setului de date
set_date = pd.read_csv("dateIN/global_economy_2021.csv")

#print(set_date.head())
#print(set_date.info())


#ACP
# Selectare variabilele numerice
numerical_columns = [
    'Population', 'Per_Capita_GNI', 'Agriculture_Forestry_Fishing',
    'Construction', 'Exports', 'Imports', 'Transport_Communication',
    'Retail_Trade_Hospitality', 'Gross_National_Income_USD', 'Gross_Domestic_Product'
]
numerical_data = set_date[numerical_columns]

# Verificare valori lipsă
print(numerical_data.isnull().sum())

# Valorile lipsa devin media coloanei
numerical_data = numerical_data.fillna(numerical_data.mean())

# Standardizare date
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Aplicare ACP
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

#A
# 1. Distribuția variației explicată de fiecare componentă
explained_variance_ratio = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'Explained Variance (%)': pca.explained_variance_ratio_ * 100
})
print("Distribuția variației explicată:")
print(explained_variance_ratio)
explained_variance_ratio.to_csv("dateOUT/Distributia_variantei", index= False)

# 2. Scorurile componentelor principale pentru observații
scores = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(len(pca.components_))])
print("\nScorurile componentelor principale:")
print(scores.head())
scores.to_csv("dateOUT/Scoruri", index=False)

# 3. Corelațiile dintre variabilele observate și componentele principale
loadings = pd.DataFrame(pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
                        index=numerical_columns)
print("\nCorelațiile dintre variabile și componente:")
print(loadings)
loadings.to_csv("dateOUT/Corelatii", index=False)

# 4. Contribuțiile variabilelor la fiecare componentă
contributions = loadings**2
print("\nContribuțiile variabilelor la fiecare componentă:")
print(contributions)
contributions.to_csv("dateOUT/Contributii", index = False)

# 5. Cosinusurile pătratice (squared cosines) ale variabilelor
squared_cosines = contributions.div(contributions.sum(axis=1), axis=0)
print("\nCosinusurile pătratice (squared cosines):")
print(squared_cosines)
squared_cosines.to_csv("dateOUT/Cosinusuri", index=False)

# 6. Comunalitățile variabilelor
communalities = contributions.sum(axis=1)
print("\nComunalitățile variabilelor:")
print(communalities)
communalities.to_csv("dateOUT/Comunalitati", index = False)

#B
#Scree plot componente criterii de selectie

# Calcularea variației cumulative explicate
explained_variance_ratio['Cumulative Explained Variance (%)'] = explained_variance_ratio['Explained Variance (%)'].cumsum()

# Graficul cu variația individuală, cumulativă și criteriile de selecție
plt.figure(figsize=(12, 8))

# Varianta explicată individual
sns.barplot(
    x=explained_variance_ratio['Component'],
    y=explained_variance_ratio['Explained Variance (%)'],
    color='skyblue',
    label='Variație individuală explicată'
)

# Varianta cumulativă explicată
plt.plot(
    explained_variance_ratio['Component'],
    explained_variance_ratio['Cumulative Explained Variance (%)'],
    marker='o', color='darkblue', label='Variație cumulativă explicată'
)

# Evidențierea criteriilor de selecție
plt.axhline(y=1/len(numerical_columns)*100, color='red', linestyle='--', label='Criteriul Kaiser (1/număr de variabile)')
plt.axhline(y=5, color='green', linestyle='--', label='Varianta minimă (5%)')
plt.axvline(x=2.5, color='orange', linestyle='--', label='Criteriul Cattell (inflecție)')  # Personalizați poziția pe baza datelor

# Setări suplimentare pentru grafic
plt.title('Distribuția variației explicată și criteriile de selecție a componentelor')
plt.xlabel('Componente principale')
plt.ylabel('Variație explicată (%)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Salvarea graficului
plt.savefig("dateOUT/Scree_plot_detaliat.png")
plt.show()


# Graficul scorurilor componentelor principale
plt.figure(figsize=(10, 6))
plt.scatter(scores['PC1'], scores['PC2'], alpha=0.7, color='purple')
plt.title('Graficul scorurilor componentelor principale (PC1 vs PC2)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("dateOUT/Plot_scoruri.png")
plt.show()

# Graficul corelațiilor dintre variabilele observate și componentele principale
plt.figure(figsize=(12, 8))
for i in range(len(loadings)):
    plt.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1], color='blue', alpha=0.8)
    plt.text(loadings.iloc[i, 0]*1.1, loadings.iloc[i, 1]*1.1, loadings.index[i], color='darkblue', fontsize=12)
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
plt.gca().add_artist(circle)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Corelațiile dintre variabilele observate și componentele principale (PC1 vs PC2)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("dateOUT/Plot_corelatii.png")
plt.show()

# Corelograma comunalităților
plt.figure(figsize=(10, 8))
sns.heatmap(communalities.to_frame(name='Comunalitate').T, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Valoare Comunalitate'})
plt.title('Corelograma Comunalităților')
plt.tight_layout()
plt.savefig("dateOUT/Corelograma_comunalitati.png")
plt.show()


#Analiza factoriala
# Testul Bartlett de sfericitate
chi_square_value, p_value = bartlett(*[numerical_data[col] for col in numerical_columns])

print("Testul Bartlett de sfericitate:")
print(f"Chi-square: {chi_square_value:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Rezultat: Datele sunt potrivite pentru analiza factorială (se respinge ipoteza nulă).")
else:
    print("Rezultat: Datele NU sunt potrivite pentru analiza factorială (nu se respinge ipoteza nulă).")

#Indicii KMO
# Calcularea indicelui KMO
kmo_all, kmo_model = calculate_kmo(numerical_data)

# Crearea unui tabel cu indicii KMO
kmo_table = pd.DataFrame({
    'Variable': numerical_data.columns,
    'KMO Index': kmo_all
})

# Afisarea tabelului cu indicii KMO
print("\nIndicele KMO pentru fiecare variabilă:")
print(kmo_table)

# Salvarea rezultatelor într-un fișier CSV
kmo_table.to_csv("dateOUT/Indecsi_KMO.csv", index=False)

#Tabel varianta factori
fa = FactorAnalyzer(n_factors=2, rotation=None)
fa.fit(scaled_data)

# Calcularea varianței explicate pentru fiecare factor
eigenvalues = fa.get_eigenvalues()[0]  # Valorile proprii
total_variance = sum(eigenvalues)  # Suma valorilor proprii
explained_variance = eigenvalues / total_variance * 100  # Procentul de varianță explicat de fiecare factor

# Crearea unui DataFrame pentru varianța explicată
variance_factors = pd.DataFrame({
    'Factor': [f'Factor{i+1}' for i in range(len(explained_variance))],
    'Explained Variance (%)': explained_variance
})

print("\nVarianța explicată de fiecare factor:")
print(variance_factors)
variance_factors.to_csv("dateOUT/Varianta_factori.csv", index=False)

#Corelatii factoriale cu si fara rotatie
#Analiza factorială cu rotație (varimax)
fa_rotated = FactorAnalyzer(n_factors=2, rotation='varimax')
fa_rotated.fit(scaled_data)

#Corelațiile între variabilele observate și factori (corelații factoriale) fără rotație
loadings_no_rotation = pd.DataFrame(fa.loadings_, columns=[f'Factor{i+1}' for i in range(fa.loadings_.shape[1])], index=numerical_columns)
print("\nCorelațiile factoriale fără rotație:")
print(loadings_no_rotation)
loadings_no_rotation.to_csv("dateOUT/Corelatii_factoriale_fara_rotatie.csv", index=False)

#Corelațiile între variabilele observate și factori (corelații factoriale) cu rotație
loadings_rotated = pd.DataFrame(fa_rotated.loadings_, columns=[f'Factor{i+1}' for i in range(fa_rotated.loadings_.shape[1])], index=numerical_columns)
print("\nCorelațiile factoriale cu rotație:")
print(loadings_rotated)
loadings_rotated.to_csv("dateOUT/Corelatii_factoriale_cu_rotatie.csv", index=False)

#Scoruri factoriale fără rotație
scores_no_rotation = fa.transform(scaled_data)
scores_no_rotation_df = pd.DataFrame(scores_no_rotation, columns=[f'Factor{i+1}' for i in range(scores_no_rotation.shape[1])])
print("\nScoruri factoriale fără rotație:")
print(scores_no_rotation_df.head())
scores_no_rotation_df.to_csv("dateOUT/Scoruri_factoriale_fata_rotatie.csv", index=False)

#Scoruri factoriale cu rotație
scores_rotated = fa_rotated.transform(scaled_data)
scores_rotated_df = pd.DataFrame(scores_rotated, columns=[f'Factor{i+1}' for i in range(scores_rotated.shape[1])])
print("\nScoruri factoriale cu rotație:")
print(scores_rotated_df.head())
scores_rotated_df.to_csv("dateOUT/Scoruri_factoriale_cu_rotatie.csv", index=False)

#Comunalitățile pentru analiza factorilor
communalities_factor = pd.DataFrame(fa_rotated.get_communalities(), index=numerical_columns, columns=['Comunalitate'])
print("\nComunalitățile pentru analiza factorilor:")
print(communalities_factor)
communalities_factor.to_csv("dateOUT/Comunalitati_factoriale.csv", index=False)

# Corelogramă Indecși KMO
plt.figure(figsize=(10, 8))
sns.heatmap(kmo_table.set_index('Variable').T, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'KMO Index Value'})
plt.title('Corelograma Indecșilor KMO')
plt.tight_layout()
plt.savefig("dateOUT/Corelograma_KMO.png")
plt.show()

# Graficul corelațiilor factoriale fără rotație
plt.figure(figsize=(12, 8))
for i in range(len(loadings_no_rotation)):
    plt.arrow(0, 0, loadings_no_rotation.iloc[i, 0], loadings_no_rotation.iloc[i, 1], color='blue', alpha=0.8)
    plt.text(loadings_no_rotation.iloc[i, 0] * 1.1, loadings_no_rotation.iloc[i, 1] * 1.1, loadings_no_rotation.index[i], color='darkblue', fontsize=12)
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
plt.gca().add_artist(circle)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Corelațiile factoriale fără rotație')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("dateOUT/Corelatii_factoriale_fara_rotatie.png")
plt.show()

# Graficul corelațiilor factoriale cu rotație
plt.figure(figsize=(12, 8))
for i in range(len(loadings_rotated)):
    plt.arrow(0, 0, loadings_rotated.iloc[i, 0], loadings_rotated.iloc[i, 1], color='green', alpha=0.8)
    plt.text(loadings_rotated.iloc[i, 0] * 1.1, loadings_rotated.iloc[i, 1] * 1.1, loadings_rotated.index[i], color='darkgreen', fontsize=12)
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
plt.gca().add_artist(circle)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Corelațiile factoriale cu rotație')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("dateOUT/Corelatii_factoriale_cu_rotatie.png")
plt.show()

# Graficul scorurilor factoriale fără rotație
plt.figure(figsize=(10, 6))
plt.scatter(scores_no_rotation_df['Factor1'], scores_no_rotation_df['Factor2'], alpha=0.7, color='purple')
plt.title('Scoruri factoriale fără rotație (Factor 1 vs Factor 2)')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("dateOUT/Plot_scoruri_fara_rotatie.png")
plt.show()

# Graficul scorurilor factoriale cu rotație
plt.figure(figsize=(10, 6))
plt.scatter(scores_rotated_df['Factor1'], scores_rotated_df['Factor2'], alpha=0.7, color='orange')
plt.title('Scoruri factoriale cu rotație (Factor 1 vs Factor 2)')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("dateOUT/Plot_scoruri_cu_rotatie.png")
plt.show()

# Corelograma comunalităților
plt.figure(figsize=(10, 8))
sns.heatmap(communalities_factor.T, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Comunalitate'})
plt.title('Corelograma Comunalităților Factoriale')
plt.tight_layout()
plt.savefig("dateOUT/Corelograma_comunalitati_factoriale.png")
plt.show()
