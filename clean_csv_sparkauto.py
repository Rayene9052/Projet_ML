import pandas as pd

# Charger le CSV
df = pd.read_csv("sparkauto.csv")

# Remplacer les valeurs manquantes par 0 (ou une autre valeur si tu préfères)
df['Puissance_fiscale'] = df['Puissance_fiscale'].fillna(0)

# Convertir en entier
df['Puissance_fiscale'] = df['Puissance_fiscale'].astype(int)

# Sauvegarder le CSV mis à jour
df.to_csv("sparkauto_fixed.csv", index=False)
