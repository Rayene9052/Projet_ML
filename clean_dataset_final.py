import pandas as pd
import numpy as np

# Charger le dataset fusionné
df = pd.read_csv("cars_dataset.csv")

# -----------------------------
# 1. Supprimer les doublons
# -----------------------------
df.drop_duplicates(subset=['Marque', 'Modèle', 'Année', 'Kilométrage', 'Carburant', 'Boîte_vitesse', 'Puissance_fiscale', 'Prix'], 
                   keep='first', inplace=True)

# -----------------------------
# 2. Nettoyer et convertir les colonnes
# -----------------------------
# Colonnes numériques
numeric_cols = ['Année', 'Kilométrage', 'Puissance_fiscale', 'Nombre_portes', 'Prix']

for col in numeric_cols:
    if col in df.columns:
        # Supprimer tout ce qui n'est pas un chiffre
        df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        # Convertir en float d'abord pour gérer les NaN et décimales
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Colonnes textuelles
text_cols = ['Marque', 'Modèle', 'Carburant', 'Boîte_vitesse', 'Etat_generale']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan})

# -----------------------------
# 3. Gérer les valeurs manquantes
# -----------------------------
# Nombre de portes → 5 par défaut si manquant
df['Nombre_portes'] = df['Nombre_portes'].fillna(5).astype(int)

# Kilométrage → remplir avec 0 si manquant
df['Kilométrage'] = df['Kilométrage'].fillna(0).astype(int)

# Puissance fiscale → remplir NaN avec 0 puis convertir en int
df['Puissance_fiscale'] = df['Puissance_fiscale'].fillna(0).astype(int)

# Prix → remplir les NaN avec la moyenne par Marque et Modèle
def fill_price(row):
    if pd.notna(row['Prix']) and row['Prix'] > 0:
        return row['Prix']
    mask = (df['Marque'] == row['Marque']) & (df['Modèle'] == row['Modèle']) & (df['Prix'].notna())
    mean_price = df.loc[mask, 'Prix'].mean()
    return mean_price if not np.isnan(mean_price) else 0

df['Prix'] = df.apply(fill_price, axis=1)

# Corriger les prix trop bas (<1000) en multipliant par 1000
df.loc[df['Prix'] < 1000, 'Prix'] = df.loc[df['Prix'] < 1000, 'Prix'] * 1000

df['Prix'] = df['Prix'].astype(int)

# -----------------------------
# 4. Supprimer les lignes avec trop de valeurs manquantes
# -----------------------------
df = df.dropna(thresh=5)  # garder les lignes avec au moins 5 valeurs non nulles

# -----------------------------
# 5. Harmoniser la casse
# -----------------------------
df['Marque'] = df['Marque'].str.title()
df['Modèle'] = df['Modèle'].str.title()
df['Carburant'] = df['Carburant'].str.title()
df['Boîte_vitesse'] = df['Boîte_vitesse'].str.title()
df['Etat_generale'] = df['Etat_generale'].str.title()

# -----------------------------
# 6. Sauvegarder le dataset nettoyé
# -----------------------------
df.to_csv("merged_cars_dataset_cleaned.csv", index=False)

print(f"Nettoyage terminé : {len(df)} lignes, {len(df.columns)} colonnes")
