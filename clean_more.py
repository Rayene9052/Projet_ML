import pandas as pd

# Charger le dataset
df = pd.read_csv("merged_cars_dataset_cleaned.csv")

# ======================================
# 1️⃣ Nettoyage Carburant
# ======================================

carburant_map = {
    # Hybrides regroupés
    "Hybride Rechargeable Essence": "Hybride",
    "Hybride Essence": "Hybride",
    "Hybride": "Hybride",
    "Hybride Diesel": "Hybride",
    "Hybride Rechargeable Diesel": "Hybride",
    
    # Variantes peu fréquentes regroupées
    "Gpl": "Autre",
    "Lpg": "Autre",
    "Micro-Hybride": "Autre",
    "Autre": "Autre",
    
    # Electrique standardisé
    "Électrique": "Electrique"
}

df["Carburant"] = df["Carburant"].replace(carburant_map)
df["Carburant"] = df["Carburant"].fillna("Autre")  # Remplacer NaN par "Autre"

# ======================================
# 2️⃣ Nettoyage Etat_generale
# Objectif : seulement "1ère main" et "Occasion"
# ======================================

etat_map = {
    "Trés Bon Etat": "1ère main",
    # Variantes 1ère main
    "1 Ere Main": "1ère main",
    "1Ere Main": "1ère main",
    "1Ère Main": "1ère main",
    "1ère main": "1ère main",

    # Variantes 2ème main -> Occasion
    "2Ème Main": "Occasion",
    "2 Eme Main": "Occasion",
    "2Eme Main": "Occasion",

    # Occasion déjà propre
    "Occasion": "Occasion",

    # Hybride Rechargeable dans Etat -> Occasion
    "Hybride Rechargeable": "Occasion"
}

df["Etat_generale"] = df["Etat_generale"].replace(etat_map)

# ======================================
# 3️⃣ Drop lignes avec valeurs manquantes critiques
# ======================================

# Colonnes critiques
colonnes_critique = ["Année", "Boîte_vitesse"]

# Supprimer lignes où Année ou Boîte_vitesse est manquante
df = df.dropna(subset=colonnes_critique)

# ======================================
# 4️⃣ Supprimer lignes avec Carburant "Autre"
# ======================================

df = df[df["Carburant"] != "Autre"]

# ======================================
# 5️⃣ Affichage des informations finales
# ======================================

print("Nombre de lignes après nettoyage :", len(df))

print("\nValeurs uniques Carburant :")
print(df["Carburant"].unique())

print("\nValeurs uniques Etat_generale :")
print(df["Etat_generale"].unique())

# Colonnes avec valeurs manquantes (après nettoyage)
missing_per_column = df.isna().sum()
missing_per_column = missing_per_column[missing_per_column > 0]
print("\nColonnes encore avec valeurs manquantes :")
print(missing_per_column)

# ======================================
# 6️⃣ Sauvegarde du dataset nettoyé
# ======================================

df.to_csv("cars_dataset_cleaned_more.csv", index=False)
