import pandas as pd

# Charger sparkauto
df_spark = pd.read_csv("sparkauto.csv")

# Ajouter Nombre_portes vide
df_spark['Nombre_portes'] = pd.NA  # ou une valeur par défaut comme 0

# Maintenant merge sera homogène
