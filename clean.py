import pandas as pd

# Charger sparkauto
df_spark = pd.read_csv("sparkauto.csv")

# Créer une colonne vide
nombre_portes_col = pd.Series([pd.NA] * len(df_spark), name='Nombre_portes')

# Insérer comme 8ème colonne (index 7)
df_spark.insert(7, 'Nombre_portes', nombre_portes_col)

# Sauvegarder le CSV mis à jour
df_spark.to_csv("sparkauto_fixed.csv", index=False)

print("Colonne 'Nombre_portes' ajoutée avec succès !")
