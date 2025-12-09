import pandas as pd

df = pd.read_csv("cars_dataset_cleaned_more.csv")

before = len(df)

# Supprimer prix = 0 ou prix null
df = df[df['Prix'].notna()]
df = df[df['Prix'] > 0]

# Supprimer les valeurs irréalistes ( > 1 million DT )
df = df[df['Prix'] <= 1_000_000]

after = len(df)

print(f"Rows before: {before}")
print(f"Rows after : {after}")
print(f"Supprimés   : {before-after}")

# Sauvegarde
df.to_csv("cars_dataset_cleaned_more_fixed.csv", index=False)

print("✔ prices cleaned & file saved.")
