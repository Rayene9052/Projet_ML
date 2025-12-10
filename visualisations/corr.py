import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
df = pd.read_csv("cars_dataset_cleaned_more_final.csv")

# Filtrer prix réaliste
df_filtered = df[(df['Prix'] >= 5000) & (df['Prix'] <= 800000)]

# Sélection des variables numériques
num_cols = ['Année', 'Kilométrage', 'Puissance_fiscale', 'Nombre_portes', 'Prix']
df_num = df_filtered[num_cols]

# Calcul de la matrice de corrélation
corr_matrix = df_num.corr()

# Affichage
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Matrice de corrélation des variables numériques")
plt.tight_layout()
plt.show()
