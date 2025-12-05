# 🚗 Car Price Prediction — Full Project Guide

> README prêt à pusher sur GitHub — contenu en français, organisé et avec tous les blocs de code Python en triple backticks.  
> Copie-colle directement ce fichier dans `README.md`.

---

## Table des matières
1. [Vue d’ensemble du workflow](#1-vue-densemble-du-workflow)  
2. [Collecte de données — stratégie concrète](#2-collecte-de-données--stratégie-concrète)  
3. [Champs (features) à extraire](#3-champs-features-à-extraire)  
4. [Exemple de scraping (BeautifulSoup)](#4-exemple-de-scraping-beautifulsoup)  
5. [Nettoyage & préparation (pandas)](#5-nettoyage--préparation-pandas)  
6. [Feature engineering](#6-feature-engineering)  
7. [Modélisation — pipeline exemple (scikit-learn)](#7-modélisation---pipeline-exemple-scikit-learn)  
8. [Évaluation — métriques & validation](#8-évaluation---métriques--validation)  
9. [Problèmes récurrents & solutions](#9-problèmes-récurrents--solutions)  
10. [Séparer les jeux de données](#10-séparer-les-jeux-de-données)  
11. [Déploiement & livraison](#11-déploiement--livraison)  
12. [Plan de travail (sprint simple)](#12-plan-de-travail-sprint-simple)  
13. [Regex utiles pour parser](#13-regex-utiles-pour-parser)  
14. [Bonnes pratiques & recommandations finales](#14-bonnes-pratiques--recommandations-finales)  

---

## 1) Vue d’ensemble du workflow

- **Définir l’objectif précis** — prédire le prix en **TND** d’une annonce voiture (occasion / neuve).  
- **Collecte de données (scraping)** — extraire annonces depuis 4–6 sites prioritaires (ex : `automobile.tn`, `sayarti.tn`, `argusautomobile.tn`, `tayara.tn`, `sparkauto.tn`, `auto-plus.tn`).  
- **Nettoyage & enrichissement** — normaliser prix, km, année, convertir texte en champs.  
- **Analyse exploratoire (EDA)** — distributions, corrélations, outliers.  
- **Feature engineering** — âge = année actuelle − année immat, `log(price)`, groupements, interactions.  
- **Modeling** — baselines (Linear/Tree), modèles robustes (RandomForest, XGBoost/CatBoost/LightGBM).  
- **Validation & métriques** — MAE, RMSE, R²; cross-validation.  
- **Déploiement** — prototype Streamlit / API (Flask/FastAPI) + Docker.  
- **Documentation & limites** — biais, prix manquants, fiabilité des annonces.

---

## 2) Collecte de données — stratégie concrète

- **Choisir 4–6 sites** (parmi tes 12) avec structure stable et annonces nombreuses. Priorité :  
  `automobile.tn`, `sayarti.tn`, `tayara.tn`, `sparkauto.tn`, `auto-plus.tn`, `argusautomobile.tn`.  
- **Échantillonnage** : prototype → **~1000–3000** annonces ; modèle solide → **≥5000** annonces.  
- **Stratégie d’échantillonnage** : si site très grand, prendre N annonces par marque / par page (ex. 5–20).  
- **Électriques** : extraire tous les électriques (dataset plus petit) et les traiter séparément ou ajouter `is_electric`.  
- **Respect & éthique** : vérifier `robots.txt`, conditions d’utilisation ; throttle requests (sleep 1–3s), user-agent, pagination soignée.  
- **Problèmes à surveiller** : annonces sans prix, prix absurdes (ex. "27 malyoum"), prix en devise différente, annonces dupliquées, données dans description (parse requis).

---

## 3) Champs (features) à extraire (prioritaires)

- `price` (TND) — **cible**  
- `brand` (Marque)  
- `model` (Modèle)  
- `year` (Année mise en circulation) → `age = 2025 - year` (ou année actuelle)  
- `mileage` (Kilométrage en km)  
- `fuel` (Carburant : Essence, Diesel, Electrique, Hybride)  
- `transmission` (Boite : Auto/Manuelle)  
- `body` (Carrosserie : berline, SUV, utilitaire, bus, ambulance)  
- `power` (Puissance fiscale / réelle si dispo)  
- `engine_cc` (Cylindrée)  
- `doors`, `seats`  
- `color_ext`, `color_int` (si dispo)  
- `seller_type` (particulier, pro, concessionnaire) — souvent dans description  
- `condition` (neuf/occasion/reconditionné)  
- `location` (ville/region)  
- `images_count`, `has_warranty`, `features_list` (clim, gps…) — convertir en flags  
- `is_electric` (flag)  
- `posting_age` (temps depuis publication) si dispo  
- **Traçabilité** : garder `source_site`, `scrape_date`, `ad_id`

---

## 4) Exemple de scraping (squelette) avec BeautifulSoup

> **Remarque** : adapter les sélecteurs HTML par site. Tester toujours sur 1 page.

```python
import requests
from bs4 import BeautifulSoup
import time
import re
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CarPriceBot/1.0)"}

def parse_price(text):
    # normalize text like "27 000 DT" -> 27000
    if not text:
        return None
    t = text.replace('\xa0', ' ').replace(',', '').lower()
    m = re.search(r'(\d[\d\s]*)\s*(dt|tnd|dinars|dinar)?', t)
    if m:
        return int(m.group(1).replace(' ', ''))
    return None

def parse_mileage(text):
    if not text:
        return None
    t = text.lower().replace(' ', '')
    m = re.search(r'(\d[\d,\.]*)\s*(km|kilom)', t)
    if m:
        return int(m.group(1).replace(',', '').replace('.', ''))
    return None

def scrape_listing_page(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = []
    # exemple générique : boucle sur les cartes d'annonce
    for card in soup.select('.ad-card, .listing-item'):
        title = card.select_one('.title, .ad-title')
        price_el = card.select_one('.price')
        link_el = card.select_one('a')
        link = link_el['href'] if link_el else None
        price = parse_price(price_el.get_text() if price_el else None)
        title_text = title.get_text(strip=True) if title else ''
        results.append({'title': title_text, 'price': price, 'link': link})
    return results

def scrape_ad_details(ad_url):
    r = requests.get(ad_url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, 'html.parser')
    # Exemples : adapter sélecteurs
    title = soup.select_one('h1').get_text(strip=True) if soup.select_one('h1') else ''
    price = parse_price(soup.select_one('.price').get_text() if soup.select_one('.price') else None)
    description = soup.select_one('.description').get_text(separator=' ', strip=True) if soup.select_one('.description') else ''
    # parsing simple des caractéristiques
    specs = {}
    for row in soup.select('.specs tr'):
        cols = row.select('td')
        if len(cols) >= 2:
            key = cols[0].get_text(strip=True).lower()
            val = cols[1].get_text(strip=True)
            specs[key] = val
    return {'title': title, 'price': price, 'description': description, 'specs': specs}

# Exemple d'utilisation
if __name__ == '__main__':
    page_url = 'https://www.example.tn/voitures/page-1'
    listings = scrape_listing_page(page_url)
    df_rows = []
    for l in listings[:20]:
        if not l['link']:
            continue
        details = scrape_ad_details(l['link'])
        df_rows.append(details)
        time.sleep(1.5)  # throttle
    df = pd.DataFrame(df_rows)
    print(df.head())
```
## 5) Nettoyage & préparation (pandas)

- Normaliser prix : retirer annonces sans prix (ou garder `price_missing` flag).  
- Convertir texte → numériques : `mileage`, `engine_cc`, `year`.  
- Dates : convertir `year` → `age`.  
- Gérer missing : imputer par médiane ou utiliser modèles qui gèrent NA (CatBoost).  
- Outliers : couper prix `< 500 TND` ou `> 5 000 000 TND` selon contexte ; log-transform du prix souvent utile.  
- Feature textuelle : extraire `seller_type` par regex (mots-clés : particulier, professionnel, concessionnaire, 1ere main).

### Exemple pipeline (pandas)

```python
import numpy as np

df['price'] = df['price'].astype(float)
df = df[df['price'].notna()]   # ou garder mais marquer

df['year'] = df['specs'].apply(lambda s: int(s.get('année', 0)) if s and s.get('année') else np.nan)
df['age'] = 2025 - df['year']

df['mileage'] = df['specs'].apply(lambda s: parse_mileage(s.get('kilométrage', '')) if s else np.nan)

# fuel doit être une colonne extraite (ex: specs.get('carburant'))
df['is_electric'] = df['fuel'].str.contains('elect', case=False, na=False).astype(int)

df['log_price'] = np.log1p(df['price'])
```
## 6) Feature engineering important

- `age` (mieux que raw `year`)  
- `log_price` comme cible pour réduire skew  
- `mileage_per_year = mileage / max(1, age)`  
- `brand_popularity` = fréquence d’apparition par marque  

**Encodages :**  
- `OneHot` pour faible cardinalité  
- `Target Encoding` / embeddings pour `model` si cardinalité haute  

**Autres features :**  
- `has_images` / `images_count` — annonces avec images valent souvent plus  
- `seller_is_dealer` flag  
- `is_electric` séparé ou interaction `brand * is_electric`

---

## 7) Modélisation — pipeline exemple (scikit-learn)

- Baselines : `LinearRegression`, `Ridge`, `Lasso`  
- Arbres : `RandomForestRegressor`, `GradientBoostingRegressor`  
- Modèles rapides et performants : `XGBoost`, `LightGBM`, `CatBoost` (CatBoost gère natif les catégoriques & NA)

### Exemple minimal (scikit-learn pipeline)

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

num_cols = ['age','mileage','engine_cc','power']
cat_cols = ['brand','model','fuel','transmission','body','seller_type']

preproc = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),
])

pipe = Pipeline([
    ('pre', preproc),
    ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

X = df[num_cols + cat_cols]
y = df['price']  # or 'log_price'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe.fit(X_train, y_train)
print("Test R2:", pipe.score(X_test, y_test))
```
# 🚗 Car Price Prediction — Sections 8 à 14 (README.md)

> Contenu prêt à coller dans ton `README.md`. Tous les blocs de code Python sont en triple backticks.

---

## 8) Évaluation — métriques & validation

- **MAE (Mean Absolute Error)** — interprétable en TND (erreur moyenne).  
- **RMSE** — pénalise fortement les gros écarts.  
- **R²** — proportion de variance expliquée par le modèle.

**Remarques :**
- Si tu entraînes sur `log(price)`, reporte les métriques sur l’échelle réelle en reconvertissant : `pred_real = np.expm1(pred_log)`.  
- **Validation** recommandée : Cross-validation (k=5 ou k=10) + hold-out final (20%) pour estimation finale.

**Analyses à produire :**
- Courbe residuals vs fitted (détecter biais non-linéaires).  
- Erreurs moyennes par `brand`, par `year` et par `segment` (citadine/SUV/etc.).  
- Distribution des erreurs (boxplots) pour détecter outliers.

---

## 9) Problèmes récurrents & solutions

- **Prix manquant** : supprimer les annonces sans prix ou marquer `price_missing` puis imputer si nécessaire.  
- **Prix erronés (typos)** : heuristiques (ex. `price < 1000` ou `price > median * 10`) + vérification manuelle d’un échantillon.  
- **Données réparties entre pages / formats différents** : architecture de scrapers modulaires (`one scraper per site`), logs d’erreurs, tests unitaires pour selecteurs.  
- **Duplicatas** : déduplication via clé `(title, price, km, year)` et/ou hash des images.  
- **Électriques** : si échantillon petit → entraîner modèle séparé ; sinon ajouter flag `is_electric`.  
- **Véhicules spéciaux** (bus, ambulances) : exclure si l’objectif est voitures particulières.

---

## 10) Séparer les jeux de données

Séparer ou tagger les subsets suivants pour entraînements et analyses :

- **Électriques vs thermiques**  
- **Occasion vs neuf**  
- **Particulier vs professionnel**  
- **Segments** : citadine / berline / SUV / utilitaire / luxe

Ceci permet de comparer performances et comportement des modèles selon sous-populations.

---

## 11) Déploiement & livraison

- **Prototype UI** : Streamlit — rapide à monter (inputs → prédiction).  
- **API** : FastAPI (recommandé) ou Flask pour exposer un endpoint `/predict` (POST JSON).  
- **Docker** : Dockerfile pour containeriser l’application (API ou Streamlit).  
- **Monitoring** : journalisation des appels et des features, stockage des prédictions vs ventes réelles pour réentraînement et dérive du modèle.

---

## 12) Plan de travail (sprint simple)

> Optionnel mais pratique pour organiser le travail.

- **Semaine 1** : Choix sites + inspection HTML + implémenter 1er scraper (2 sites) → collecter ~500 annonces.  
- **Semaine 2** : Nettoyage des données, EDA, features de base, baseline linéaire.  
- **Semaine 3** : Modèles avancés (XGBoost / CatBoost / LightGBM), hyperparam tuning.  
- **Semaine 4** : Prototype Streamlit / API FastAPI + Docker + documentation + rapport final.

---

## 13) Exemples d’expressions / regex utiles pour parser

**Prix**  
```python
m = re.search(r'(\d[\d\s]*)\s*(dt|tnd|dinar)', text, re.I)
if m:
    price = int(m.group(1).replace(' ', ''))
```
# 📌 Extras : Regex & bonnes pratiques (à coller dans README.md)

## Année

```python
import re

m = re.search(r'(\d{4})', text)
if m:
    year = int(m.group(1))  # vérifier 1980 <= year <= 2025
```
## Kilometrage
```python

import re

m = re.search(r'(\d[\d.,]*)\s*(km|kilom)', text, re.I)
if m:
    km = int(m.group(1).replace('.', '').replace(',', ''))

```
## Première main (seller_type)
```python
desc = description.lower()
if '1ère main' in desc or 'premiere main' in desc or 'première main' in desc:
    seller_type = 'first_owner'
elif 'concessionnaire' in desc or 'garantie' in desc:
    seller_type = 'dealer'
else:
    seller_type = 'private'

```
## 14) Bonnes pratiques & recommandations finales

- **Démarrer petit** : 500–1000 annonces pour prototypage.  
- **Modularité** : un scraper par site + fonctions utilitaires partagées.  
- **Traçabilité** : stocker `source_site`, `scrape_date`, `ad_id`, éventuellement `raw_html`.  
- **Tests automatisés** : assertions sur les selecteurs, tests d’intégration pour scrapers.  
- **Respect légal / éthique** : vérifier `robots.txt`, éviter le scraping agressif.  
- **Features externes** : envisager taux de change, indices économiques, coût moyen local pour améliorer le modèle.  

---

## Structure de dépôt recommandée
```
car-price-prediction/
├─ data/
│  ├─ raw/          # raw html / raw JSON
│  ├─ interim/
│  └─ processed/    # cleaned CSV / parquet
├─ notebooks/
│  └─ 01_EDA_and_modeling.ipynb
├─ src/
│  ├─ scraping/
│  │  ├─ __init__.py
│  │  ├─ automobile_tn.py
│  │  ├─ sayarti_tn.py
│  │  └─ utils.py
│  ├─ preprocessing/
│  │  └─ clean.py
│  ├─ features/
│  │  └─ build_features.py
│  ├─ models/
│  │  ├─ train.py
│  │  └─ predict.py
│  └─ api/
│     └─ app.py     # FastAPI or Streamlit app
├─ models/
│  └─ model.pkl
├─ Dockerfile
├─ requirements.txt
└─ README.md
```
