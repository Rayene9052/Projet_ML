import pandas as pd

def fix_kilometrage(df):
    # convertir Etat_generale en minuscule
    df["Etat_generale"] = df["Etat_generale"].astype(str).str.lower()
    
    # multiplier seulement *occasion*
    mask = (df["Kilométrage"] < 1000) & (df["Etat_generale"] == "occasion")
    df.loc[mask, "Kilométrage"] = df.loc[mask, "Kilométrage"] * 1000

    return df


if __name__ == "__main__":
    df = pd.read_csv("cars_dataset_cleaned_more_final.csv")

    df = fix_kilometrage(df)

    df.to_csv("cars_dataset_cleaned_more_final_v2.csv", index=False)
    print("Kilométrage corrigé ✔")
