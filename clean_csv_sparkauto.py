import pandas as pd
import numpy as np

df = pd.read_csv("sparkauto.csv")

def clean_puissance(x):
    try:
        # Extraire les chiffres si possible
        return int(float(str(x).split()[0]))
    except:
        return np.nan

df['Puissance_fiscale'] = df['Puissance_fiscale'].apply(clean_puissance)

df['Puissance_fiscale'] = df['Puissance_fiscale'].astype('Int64')

df.to_csv("sparkauto_clean.csv", index=False)
