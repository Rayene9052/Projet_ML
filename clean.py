import pandas as pd

# Charger le CSV
df = pd.read_csv("baniola_cars_features.csv")

# Retirer 'CV' et convertir en entier
df['Puissance_fiscale'] = pd.to_numeric(df['Puissance_fiscale'].str.replace('CV', '', regex=False).str.strip(), errors='coerce').fillna(0).astype(int)

# Sauvegarder le CSV mis à jour
df.to_csv("baniola_cars_features_fixed.csv", index=False)
