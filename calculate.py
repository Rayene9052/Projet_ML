import pandas as pd

# Charger le dataset
df = pd.read_csv("cars_dataset_cleaned_more_final.csv")

# S'assurer que les colonnes sont numériques
df['PUISSANCE_FISCALE'] = pd.to_numeric(df['PUISSANCE_FISCALE'], errors='coerce')
df['KILOMÉTRAGE'] = pd.to_numeric(df['KILOMÉTRAGE'], errors='coerce')

# Trouver les valeurs maximales
max_puissance = df['PUISSANCE_FISCALE'].max()
max_kilometrage = df['KILOMÉTRAGE'].max()

print(f"Valeur maximale de PUISSANCE_FISCALE : {max_puissance}")
print(f"Valeur maximale de KILOMÉTRAGE : {max_kilometrage}")
