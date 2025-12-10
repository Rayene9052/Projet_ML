import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Charger le dataset
df = pd.read_csv("cars_dataset_cleaned_more_final.csv")

# -----------------------------
# Variables numériques
# -----------------------------
num_cols = ['Année', 'Kilométrage', 'Puissance_fiscale', 'Nombre_portes', 'Prix']

for col in num_cols:
    plt.figure(figsize=(10,6))
    if col == 'Kilométrage':
        # Zoom sur 0-400000 km avec bins plus précis
        bins = np.linspace(0, 400000, 50)
        sns.histplot(df[col].dropna(), bins=bins, kde=False, color='darkgreen')
        plt.xlim(0, 400000)
    elif col == 'Année':
        # Bins tous les 5 ans et afficher toutes les années sur l'axe X
        min_year = int(df['Année'].min())
        max_year = int(df['Année'].max())
        bins = list(range(min_year - min_year%5, max_year + 5, 5))
        sns.histplot(df[col].dropna(), bins=bins, kde=False, color='darkblue')
        plt.xticks(bins, rotation=45)
    elif col == 'Puissance_fiscale':
        bins = sorted(df[col].dropna().unique())
        sns.histplot(df[col].dropna(), bins=bins, kde=False, color='purple')
        plt.xticks(bins)
    elif col == 'Prix':
    # Limite entre 5000 et 800000 DT
        filtered = df[(df['Prix'] >= 5000) & (df['Prix'] <= 800000)]
        bins = np.linspace(5000, 800000, 60)  # plus de bins pour plus de détails
        sns.histplot(filtered['Prix'], bins=bins, kde=False, color='darkred')
        plt.xlim(5000, 800000)
        # Afficher plus de ticks avec intervalle de 25000 DT
        plt.xticks(np.arange(5000, 800001, 25000), rotation=45)

    else:
        sns.histplot(df[col].dropna(), bins=30, kde=False, color='orange')
    plt.title(f'Distribution de {col}', fontsize=16)
    plt.xlabel(col)
    plt.ylabel('Nombre de véhicules')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Variables catégorielles
# -----------------------------
cat_cols = ['Marque', 'Modèle', 'Carburant', 'Boîte_vitesse', 'Etat_generale']

for col in cat_cols:
    plt.figure(figsize=(12,6))
    if col == 'Modèle':
        top_n = 50
        counts = df[col].value_counts().head(top_n)
        sns.barplot(x=counts.index, y=counts.values, palette='dark:#5A9_r')
        plt.title(f'Top {top_n} modèles les plus fréquents', fontsize=16)
        plt.xlabel(col)
        plt.ylabel('Nombre de véhicules')
        plt.xticks(rotation=45, ha='right')
    else:
        counts = df[col].value_counts()
        sns.barplot(x=counts.index, y=counts.values, palette='Set2')
        plt.title(f'Distribution de {col}', fontsize=16)
        plt.xlabel(col)
        plt.ylabel('Nombre de véhicules')
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
