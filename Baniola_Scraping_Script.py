from bs4 import BeautifulSoup as bs
import pandas as pd
import time
import requests
import json

BASE_URL = "https://baniola.tn/voitures"  # ou .../voitures/regions/Tunis
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BaniolaScraper/1.0)"
}

def get_items_from_list_page(page_idx: int):
    """
    page_idx = 0 -> https://baniola.tn/voitures
    page_idx = 1 -> https://baniola.tn/voitures/50
    page_idx = 2 -> https://baniola.tn/voitures/100
    etc.
    """
    if page_idx == 0:
        url = BASE_URL
    else:
        offset = page_idx * 50
        url = f"{BASE_URL}/{offset}"

    print(f"Requête liste page_idx={page_idx} -> {url}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"Erreur HTTP/réseau sur {url}: {e}")
        return []

    soup = bs(r.content, "html.parser")
    script_tag = soup.find("script", type="application/ld+json")
    if not script_tag:
        print(f"page_idx {page_idx} -> pas de JSON-LD trouvé, on s'arrête.")
        return []

    try:
        data = json.loads(script_tag.string)
    except json.JSONDecodeError as e:
        print(f"page_idx {page_idx} -> JSON-LD invalide: {e}")
        return []

    items = data.get("itemListElement", [])
    return [elt["item"] for elt in items]

# ==========================
# 1) Récupération de toutes les URLs d'annonces
# ==========================

all_items = []
MAX_PAGE_IDX = 70  # 0 -> page1, 1 -> page2 (50), etc. adapte si besoin

for page_idx in range(0, MAX_PAGE_IDX + 1):
    items = get_items_from_list_page(page_idx)
    if not items:
        print(f"Aucune annonce trouvée pour page_idx {page_idx}, arrêt pagination.")
        break

    print(f"page_idx {page_idx}: {len(items)} annonces trouvées.")
    all_items.extend(items)
    time.sleep(1)  # pour ne pas surcharger le site

print(f"Total annonces collectées: {len(all_items)}")

all_urls = [it["url"] for it in all_items]

# ==========================
# 2) Parsing des caractéristiques sur chaque page détail
# ==========================

def parse_specs_from_soup(soup: bs) -> dict:
    specs = {
        "Marque": None,
        "Modèle": None,
        "Année": None,
        "Kilométrage": None,
        "Carburant": None,
        "Boîte vitesse": None,
        "Puissance fiscale": None,
        "Nombre de portes": None,
    }

    for line in soup.select(".specs-list .spec-line"):
        name_tag = line.select_one(".spec-name")
        data_tag = line.select_one(".spec-data")
        if not name_tag or not data_tag:
            continue

        label = name_tag.get_text(strip=True)
        value = data_tag.get_text(strip=True)

        if label in specs:
            specs[label] = value

    return specs

def scrape_car(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    soup = bs(r.text, "html.parser")

    # 1) Caractéristiques (bloc HTML)
    specs = parse_specs_from_soup(soup)

    # 2) Prix (JSON-LD Car)
    price = None
    script_tag = soup.find("script", type="application/ld+json")
    if script_tag:
        try:
            data = json.loads(script_tag.string)
            offers = data.get("offers", {})
            price = offers.get("price")
        except Exception:
            pass

    return {
        "Marque": specs["Marque"],
        "Modèle": specs["Modèle"],
        "Année": specs["Année"],
        "Kilométrage": specs["Kilométrage"],
        "Carburant": specs["Carburant"],
        "Boîte_vitesse": specs["Boîte vitesse"],
        "Puissance_fiscale": specs["Puissance fiscale"],
        "Nombre_portes": specs["Nombre de portes"],
        "Etat_generale": "Occasion",
        "Prix": price,
    }

rows = []
for i, url in enumerate(all_urls, start=1):
    print(f"[{i}/{len(all_urls)}] {url}")
    try:
        rows.append(scrape_car(url))
    except Exception as e:
        print("Erreur sur", url, ":", e)
    time.sleep(1)  # pour respecter le site

df = pd.DataFrame(rows)
df.to_csv("baniola_cars_features.csv", index=False, encoding="utf-8-sig")
print("Fichier baniola_cars_features.csv créé.")
