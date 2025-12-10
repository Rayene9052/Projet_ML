import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor

# ======================================================
# 1. CHARGER LE MODÈLE ET INFOS
# ======================================================
print("="*70)
print("🚗 SYSTÈME DE PRÉDICTION DE PRIX DE VOITURES EN TUNISIE")
print("="*70)
print("\n📦 Chargement du modèle...\n")

try:
    final_model = CatBoostRegressor()
    final_model.load_model("car_price_catboost_final.cbm")
    
    with open("dataset_info.pkl", "rb") as f:
        dataset_info = pickle.load(f)
    
    print("✅ Modèle chargé avec succès!")
    print("✅ Données du dataset chargées!")
    
except FileNotFoundError as e:
    print(f"❌ Erreur: {e}")
    print("Assurez-vous que 'car_price_catboost_final.cbm' et 'dataset_info.pkl' existent")
    exit()

# ======================================================
# 2. VARIABLES GLOBALES
# ======================================================
features = dataset_info['features']
cat_cols = dataset_info['cat_cols']

def simplifier_carburant(c):
    """Normaliser le type de carburant"""
    if pd.isna(c):
        return "Thermique"
    c_str = str(c).lower()
    if "elect" in c_str:
        return "Electrique"
    if "hybride" in c_str:
        return "Hybride"
    return "Thermique"

# ======================================================
# 3. FONCTION DE PRÉDICTION
# ======================================================
def predict_car_price(new_car_df):
    """Prédire le prix d'une voiture"""
    df_new = new_car_df.copy()
    
    try:
        # Feature engineering
        df_new["Age"] = 2025 - df_new["Année"]
        df_new["Age"] = df_new["Age"].clip(lower=0, upper=50)
        
        df_new["Usure_km_par_an"] = df_new["Kilométrage"] / df_new["Age"].replace(0, 1)
        df_new["Usure_km_par_an"] = df_new["Usure_km_par_an"].clip(upper=50000)
        
        df_new["Log_Kilometre"] = np.log1p(df_new["Kilométrage"])
        df_new["Log_Puissance"] = np.log1p(df_new["Puissance_fiscale"])
        
        marque = df_new["Marque"].values[0]
        modele = df_new["Modèle"].values[0]
        carburant = df_new["Carburant"].values[0]
        
        df_new["Prix_moy_marque"] = dataset_info['prix_moy_marque'].get(marque, dataset_info['prix_global_mean'])
        df_new["Prix_median_marque"] = df_new["Prix_moy_marque"]
        df_new["Prix_std_marque"] = 0
        df_new["Count_marque"] = dataset_info['marque_count'].get(marque, 1)
        
        df_new["Prix_moy_modele"] = dataset_info['prix_moy_modele'].get(modele, dataset_info['prix_global_mean'])
        df_new["Prix_median_modele"] = df_new["Prix_moy_modele"]
        df_new["Count_modele"] = dataset_info['modele_count'].get(modele, 1)
        
        key_fuel = (marque, carburant)
        df_new["Prix_moy_marque_fuel"] = dataset_info['prix_moy_marque_fuel'].get(key_fuel, dataset_info['prix_global_mean'])
        
        df_new["Puiss_Par_Age"] = df_new["Puissance_fiscale"] / df_new["Age"].replace(0, 1)
        df_new["Puiss_Par_Km"] = df_new["Puissance_fiscale"] / (df_new["Kilométrage"] + 1)
        df_new["Prix_per_km"] = dataset_info['prix_global_mean'] / (df_new["Kilométrage"] + 1)
        
        df_new["Carburant_simplifié"] = df_new["Carburant"].apply(simplifier_carburant)
        
        df_new["Km_category"] = pd.cut(
            df_new["Kilométrage"], 
            bins=[0, 50000, 100000, 150000, 250000, float('inf')],
            labels=['Très_faible', 'Faible', 'Moyen', 'Élevé', 'Très_élevé']
        )
        
        df_new["Age_category"] = pd.cut(
            df_new["Age"], 
            bins=[0, 3, 7, 12, 20, 100],
            labels=['Neuf', 'Récent', 'Moyen_age', 'Ancien', 'Très_ancien']
        )
        
        for col in cat_cols:
            if col in df_new.columns:
                df_new[col] = df_new[col].astype(str)
        
        X_new = df_new[features]
        preds_log = final_model.predict(X_new)
        preds = np.expm1(preds_log)
        
        return float(preds[0])
    
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        return None

# ======================================================
# 4. FONCTION D'AFFICHAGE RÉSULTATS
# ======================================================
def afficher_resultat(caracteristiques, prix_predit):
    """Afficher les résultats de manière lisible"""
    print("\n" + "="*70)
    print("📋 CARACTÉRISTIQUES SAISIES:")
    print("="*70)
    for key, value in caracteristiques.items():
        if isinstance(value, (int, float)):
            if key == "Kilométrage":
                print(f"   • {key}: {value:,} km")
            elif key == "Puissance_fiscale":
                print(f"   • {key}: {value} CV")
            else:
                print(f"   • {key}: {value}")
        else:
            print(f"   • {key}: {value}")
    
    if prix_predit is not None:
        print("\n" + "="*70)
        print("💰 RÉSULTAT DE LA PRÉDICTION:")
        print("="*70)
        print(f"   Prix estimé: {prix_predit:,.2f} TND")
        print(f"   Fourchette (±10%): {prix_predit*0.9:,.0f} - {prix_predit*1.1:,.0f} TND")
    else:
        print("\n❌ Erreur lors de la prédiction")
    
    print("="*70)

# ======================================================
# 5. INTERFACE INTERACTIVE
# ======================================================
print("\n" + "="*70)
print("🧪 PRÉDICTION DYNAMIQUE DE PRIX")
print("="*70)

while True:
    print("\n")
    
    # ===== ENTRÉES CATÉGORIQUES (choix parmi liste) =====
    
    # Marque
    print(f"\n📌 Marques disponibles ({len(dataset_info['marques'])} total):")
    for i, marque in enumerate(dataset_info['marques'][:10], 1):
        print(f"   {i}. {marque}", end="  ")
    print("\n   ...")
    
    while True:
        marque = input("\n👉 Entrez la Marque (ou liste pour voir plus): ").strip()
        if marque.lower() == "liste":
            print("\nListe complète des marques:")
            for i, m in enumerate(dataset_info['marques'], 1):
                print(f"   {i}. {m}")
            continue
        if marque in dataset_info['marques']:
            break
        print(f"❌ Marque non trouvée. Veuillez entrer une marque existante.")
    
    # Modèle
    print(f"\n📌 Modèles disponibles ({len(dataset_info['modeles'])} total):")
    for i, modele in enumerate(dataset_info['modeles'][:10], 1):
        print(f"   {i}. {modele}", end="  ")
    print("\n   ...")
    
    while True:
        modele = input("\n👉 Entrez le Modèle (ou liste pour voir plus): ").strip()
        if modele.lower() == "liste":
            print("\nListe complète des modèles:")
            for i, m in enumerate(dataset_info['modeles'], 1):
                print(f"   {i}. {m}")
            continue
        if modele in dataset_info['modeles']:
            break
        print(f"❌ Modèle non trouvé. Veuillez entrer un modèle existant.")
    
    # Carburant
    print(f"\n📌 Carburants disponibles:")
    for i, carburant in enumerate(dataset_info['carburants'], 1):
        print(f"   {i}. {carburant}")
    
    while True:
        carburant = input("\n👉 Entrez le Carburant (1-{}) ou nom: ".format(len(dataset_info['carburants']))).strip()
        try:
            if carburant.isdigit() and 1 <= int(carburant) <= len(dataset_info['carburants']):
                carburant = dataset_info['carburants'][int(carburant) - 1]
            if carburant in dataset_info['carburants']:
                break
        except:
            pass
        print(f"❌ Carburant invalide. Veuillez choisir parmi: {', '.join(dataset_info['carburants'])}")
    
    # Boîte vitesse
    print(f"\n📌 Boîtes de vitesse disponibles:")
    for i, boite in enumerate(dataset_info['boites'], 1):
        print(f"   {i}. {boite}")
    
    while True:
        boite = input("\n👉 Entrez la Boîte vitesse (1-{}) ou nom: ".format(len(dataset_info['boites']))).strip()
        try:
            if boite.isdigit() and 1 <= int(boite) <= len(dataset_info['boites']):
                boite = dataset_info['boites'][int(boite) - 1]
            if boite in dataset_info['boites']:
                break
        except:
            pass
        print(f"❌ Boîte vitesse invalide. Veuillez choisir parmi: {', '.join(dataset_info['boites'])}")
    
    # État général
    print(f"\n📌 États disponibles:")
    for i, etat in enumerate(dataset_info['etats'], 1):
        print(f"   {i}. {etat}")
    
    while True:
        etat = input("\n👉 Entrez l'État général (1-{}) ou nom: ".format(len(dataset_info['etats']))).strip()
        try:
            if etat.isdigit() and 1 <= int(etat) <= len(dataset_info['etats']):
                etat = dataset_info['etats'][int(etat) - 1]
            if etat in dataset_info['etats']:
                break
        except:
            pass
        print(f"❌ État invalide. Veuillez choisir parmi: {', '.join(dataset_info['etats'])}")
    
    # ===== ENTRÉES NUMÉRIQUES =====
    
    while True:
        try:
            annee = int(input("\n👉 Entrez l'Année: "))
            if 1900 <= annee <= 2025:
                break
            print("❌ L'année doit être entre 1900 et 2025")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide")
    
    while True:
        try:
            kilometrage = int(input("👉 Entrez le Kilométrage (km): "))
            if kilometrage >= 0:
                break
            print("❌ Le kilométrage doit être positif")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide")
    
    while True:
        try:
            puissance = int(input("👉 Entrez la Puissance fiscale (CV): "))
            if puissance > 0:
                break
            print("❌ La puissance doit être positive")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide")
    
    # ===== PRÉDICTION =====
    
    test_car = pd.DataFrame({
        'Marque': [marque],
        'Modèle': [modele],
        'Année': [annee],
        'Kilométrage': [kilometrage],
        'Carburant': [carburant],
        'Boîte_vitesse': [boite],
        'Puissance_fiscale': [puissance],
        'Etat_generale': [etat]
    })
    
    prix_predit = predict_car_price(test_car)
    
    afficher_resultat(
        {
            "Marque": marque,
            "Modèle": modele,
            "Année": annee,
            "Kilométrage": kilometrage,
            "Carburant": carburant,
            "Boîte_vitesse": boite,
            "Puissance_fiscale": puissance,
            "État": etat
        },
        prix_predit
    )
    
    # ===== NOUVEAU TEST =====
    
    while True:
        response = input("\n\n🔄 Voulez-vous tester une autre voiture? (oui/non): ").lower().strip()
        if response in ['non', 'n']:
            print("\n👋 Merci d'avoir utilisé le système de prédiction!")
            exit()
        elif response in ['oui', 'o']:
            break
        print("⚠️ Veuillez entrer 'oui' ou 'non'")