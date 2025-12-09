import pandas as pd

# Charger les 3 datasets
df_occasion = pd.read_csv("automobile_tn_occasion.csv")
df_spark = pd.read_csv("sparkauto.csv")  # avec Nombre_portes ajouté
df_baniola = pd.read_csv("baniola_cars_features.csv")

# Concaténer tous les datasets
df_all = pd.concat([df_occasion, df_spark, df_baniola], ignore_index=True)

# Sauvegarder le dataset fusionné
df_all.to_csv("cars_dataset.csv", index=False)

print(f"Merge terminé : {len(df_all)} lignes au total")
