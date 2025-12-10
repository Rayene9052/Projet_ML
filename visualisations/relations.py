import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
df = pd.read_csv("cars_dataset_cleaned_more_final.csv")

# Filtrage prix réaliste
df_filtered = df[(df['Prix'] >= 5000) & (df['Prix'] <= 800000)]

# -----------------------------
# 1. Prix moyen vs Marque
# -----------------------------
top_marques = df_filtered['Marque'].value_counts().head(20).index
df_marque = df_filtered[df_filtered['Marque'].isin(top_marques)]

print("\n=== Prix moyen par Marque (Top 20) ===")
print(df_marque.groupby('Marque')['Prix'].mean().sort_values(ascending=False))

# Barplot
marque_mean = df_marque.groupby('Marque')['Prix'].mean().sort_values(ascending=False)
marque_mean.plot(kind='bar', figsize=(12,6), color='skyblue')
plt.title("Prix moyen par Marque (Top 20)")
plt.ylabel("Prix moyen (DT)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# -----------------------------
# 2. Prix moyen vs Année
# -----------------------------
top_years = df_filtered['Année'].value_counts().head(20).index
df_year = df_filtered[df_filtered['Année'].isin(top_years)]

print("\n=== Prix moyen par Année (Top 20) ===")
print(df_year.groupby('Année')['Prix'].mean().sort_index())

# Barplot
year_mean = df_year.groupby('Année')['Prix'].mean().sort_index()
year_mean.plot(kind='bar', figsize=(12,6), color='lightgreen')
plt.title("Prix moyen par Année (Top 20)")
plt.ylabel("Prix moyen (DT)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Prix moyen vs Puissance Fiscale
# -----------------------------
top_puissances = df_filtered['Puissance_fiscale'].value_counts().head(20).index
df_puissance = df_filtered[df_filtered['Puissance_fiscale'].isin(top_puissances)]

print("\n=== Prix moyen par Puissance Fiscale (Top 20) ===")
print(df_puissance.groupby('Puissance_fiscale')['Prix'].mean().sort_index())

# Barplot
puissance_mean = df_puissance.groupby('Puissance_fiscale')['Prix'].mean().sort_index()
puissance_mean.plot(kind='bar', figsize=(12,6), color='salmon')
plt.title("Prix moyen par Puissance Fiscale")
plt.ylabel("Prix moyen (DT)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
