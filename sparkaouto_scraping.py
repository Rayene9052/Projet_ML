#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SparkAuto targeted scraper — extract only the requested fields (no images)
Fields: Marque, Modèle, Année, Kilométrage, Carburant, Boîte_vitesse,
        Puissance_fiscale, Nombre_portes, Etat_generale, Prix
Output: sparkauto_clean.csv (utf-8-sig)
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import time
import random
import pandas as pd
from datetime import datetime

BASE = "https://www.sparkauto.tn"
START_LISTING = BASE + "/"
MAX_PAGES = 6               # commencer petit; augmenter si OK
ADS_LIMIT = None            # None = toutes; ou int pour limiter le nombre total d'annonces
REQUEST_TIMEOUT = 12
DELAY_MIN, DELAY_MAX = 0.4, 1.0
OUTPUT_CSV = "sparkauto_clean.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SparkAutoTargetedScraper/1.0)"
}

session = requests.Session()
session.headers.update(HEADERS)

# ----- helpers -----
def safe_get(url, retries=2):
    for i in range(retries):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as e:
            if i + 1 < retries:
                time.sleep(1 + random.random())
                continue
            print("Request failed:", url, "->", e)
            return None

def clean(s):
    if s is None:
        return ""
    return " ".join(str(s).replace("\r", " ").replace("\n", " ").split()).strip()

def parse_price(text):
    if not text: 
        return None
    t = str(text)
    # priorité : nombres avec séparateur de milliers (ex: "89 000" ou "89,000" ou "89.000")
    m = re.search(r'([0-9]{1,3}(?:[ \.,\u202f][0-9]{3})+)\s*(?:dt|tnd|dinar)?', t, flags=re.I)
    if m:
        return int(re.sub(r'[^\d]', '', m.group(1)))
    m2 = re.search(r'([0-9]{3,})\s*(?:dt|tnd|dinar)?', t, flags=re.I)
    if m2:
        return int(re.sub(r'[^\d]', '', m2.group(1)))
    m3 = re.search(r'([0-9]+)', t)
    return int(m3.group(1)) if m3 else None

def parse_year(text):
    if not text: return None
    m = re.search(r'(19|20)\d{2}', str(text))
    return int(m.group(0)) if m else None

def parse_km(text):
    if not text: return None
    m = re.search(r'([0-9][0-9\.,\s\u202f]*)\s*(km|kilom)', str(text), flags=re.I)
    if m:
        return int(re.sub(r'[^\d]', '', m.group(1)))
    m2 = re.search(r'([0-9]{3,})', str(text))
    return int(re.sub(r'[^\d]', '', m2.group(1))) if m2 else None

# liste de marques courantes (heuristique pour séparer marque/modèle)
BRANDS = [
    "BMW","Mercedes","Peugeot","Renault","Volkswagen","Toyota","Hyundai","Kia","Ford","Audi","Nissan",
    "Opel","Suzuki","Dacia","Fiat","Mazda","Honda","Chevrolet","Mitsubishi","Seat","Skoda","Volvo",
    "Jaguar","Land Rover","Porsche","Jeep","Chery","Geely","MG","Tesla","Cadillac","Lada","Ssangyong","Haval","Cupra","Alfa Romeo","Citroen"
]

def extract_brand_model(title):
    t = clean(title)
    if not t:
        return "", ""
    # recherche d'une marque connue dans le titre (match case-insensitive)
    tl = t.lower()
    for b in BRANDS:
        if b.lower() in tl:
            # retirer la marque trouvée (première occurrence)
            pat = re.compile(re.escape(b), flags=re.I)
            model = pat.sub("", t, count=1).strip()
            brand = b
            # si model vide, on laisse title as model
            if model == "":
                model = t
            return brand, model
    # fallback : split by first space (heuristique)
    parts = t.split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])

# ----- extrait les liens de listing-detail sur une page de listing -----
def extract_listing_links(listing_html):
    soup = BeautifulSoup(listing_html, "html.parser")
    links = []
    # les liens d'annonces semblent contenir "listing-detail" (vu dans ton log)
    for a in soup.select("a[href*='listing-detail']"):
        href = a.get("href")
        if not href:
            continue
        full = urljoin(BASE, href)
        # normalisation : enlever parameters inutiles
        parsed = urlparse(full)
        full_norm = parsed.scheme + "://" + parsed.netloc + parsed.path
        if full_norm not in links:
            links.append(full_norm)
    return links

# ----- parse detail page ciblé, en se basant sur la structure fournie -----
def parse_listing_detail(html, url):
    soup = BeautifulSoup(html, "html.parser")

    # trouver conteneur principal : heuristique
    container = None
    for sel in ['.listing-detail', '.listing-detail-wrapper', 'main', 'article', '#content', '.product-detail', '.product']:
        container = soup.select_one(sel)
        if container:
            break
    if not container:
        container = soup.body or soup

    # titre exact : <span class="product-name ...">...</span>
    title_el = container.select_one("span.product-name") or container.select_one("h1") or container.select_one(".product-title")
    title = clean(title_el.get_text(" ", strip=True)) if title_el else ""

    # product-description (parfois contient info moteur ex "420 d")
    product_desc = ""
    desc_el = container.select_one("p.product-description")
    if desc_el:
        product_desc = clean(desc_el.get_text(" ", strip=True))

    # Etat_generale (badge, ex "1 ere Main")
    etat = ""
    # badge with class kilometrage badge also used for first owner in snippet
    badge = container.select_one(".kilometrage.badge") or container.select_one(".badge")
    if badge:
        etat = clean(badge.get_text(" ", strip=True))
    # alternative: check any span with 'Main' word
    if not etat:
        for sp in container.select("span"):
            txt = clean(sp.get_text(" ", strip=True))
            if txt and re.search(r'\bmain\b', txt, flags=re.I):
                etat = txt
                break

    # Prix : élément with class price in second-row block (voir snippet)
    price = None
    # search for parent container 'second-row' then find .price
    sec = container.select_one(".second-row") or container
    price_candidates = sec.select(".price") + sec.select(".prix") + sec.select("p.price")
    for pc in price_candidates:
        # prend le texte complet de son parent pour capter "89 ,000 DT"
        parent_text = pc.parent.get_text(" ", strip=True) if pc.parent else pc.get_text(" ", strip=True)
        price = parse_price(parent_text)
        if price:
            break
    if not price:
        # fallback: search whole container
        price = parse_price(container.get_text(" ", strip=True))

    # Gather feature blocks : each .feature contains title and content
    features = {}
    for feat in container.select(".feature"):
        # title in span.title and value in span.content (as in snippet)
        key_el = feat.select_one(".title") or feat.select_one("span.title")
        val_el = feat.select_one(".content") or feat.select_one("span.content")
        if key_el and val_el:
            k = clean(key_el.get_text(" ", strip=True))
            v = clean(val_el.get_text(" ", strip=True))
            if k:
                features[k] = v

    # autre présentation: .d-flex .title / .content pairs (fallback)
    if not features:
        for title_tag in container.select(".title"):
            try:
                parent = title_tag.parent
                value_tag = parent.select_one(".content")
                if value_tag:
                    k = clean(title_tag.get_text()); v = clean(value_tag.get_text())
                    if k: features[k] = v
            except Exception:
                continue

    # extraction par labels habituels
    kilom = features.get("Kilométrage") or features.get("Kilométrage ") or features.get("Kilométrage:")
    carburant = features.get("Carburant") or features.get("Énergie") or features.get("Fuel")
    boite = features.get("Transmission") or features.get("Boîte") or features.get("Boîte_vitesse")
    puissance = features.get("Puissance Fiscale") or features.get("Puissance Fiscale ") or features.get("Puissance")
    portes = features.get("Nombre de portes") or features.get("Portes") or features.get("Doors")
    # année peut être dans "1ère Immat." content e.g. "Novembre 2016"
    immat = features.get("1ère Immat.") or features.get("1ère Immat") or features.get("Première immatriculation") or features.get("1ère Immatriculation")
    year = parse_year(immat) if immat else None
    # si pas trouvé, essayer d'extraire année depuis titre ou product_desc
    if not year:
        year = parse_year(title) or parse_year(product_desc)

    # kilométrage numeric
    km_val = parse_km(kilom) if kilom else None
    # puissance fiscale: extraire nombre (ex "10 CV")
    pf_val = None
    if puissance:
        m = re.search(r'([0-9]{1,3})', puissance)
        if m:
            pf_val = int(m.group(1))

    # nombre de portes numeric
    portes_val = None
    if portes:
        m = re.search(r'(\d)', portes)
        if m:
            portes_val = int(m.group(1))

    # marque / modele heuristique
    brand, model = extract_brand_model(title)

    # final dict with required normalized fields
    result = {
        "Marque": brand,
        "Modèle": model,
        "Année": year,
        "Kilométrage": km_val,
        "Carburant": carburant or "",
        "Boîte_vitesse": boite or "",
        "Puissance_fiscale": pf_val,
        "Nombre_portes": portes_val,
        "Etat_generale": etat,
        "Prix": price,
        "url": url,
        "title_raw": title
    }
    return result

# ----- pipeline principal -----
def main():
    all_links = []
    scraped = []
    seen = set()

    for page in range(1, MAX_PAGES + 1):
        if page == 1:
            url = START_LISTING
        else:
            # pagination via ?page=
            if '?' in START_LISTING:
                url = START_LISTING + "&page=" + str(page)
            else:
                url = START_LISTING.rstrip('/') + "?page=" + str(page)
        print(f"[list] fetching page {page}: {url}")
        r = safe_get(url)
        if not r:
            print("  failed to fetch listing page")
            continue
        links = extract_listing_links(r.text)
        print(f"  found {len(links)} listing-detail links on page {page}")
        # ajout unique et optional limit
        for l in links:
            if l not in seen:
                seen.add(l)
                all_links.append(l)
            if ADS_LIMIT and len(all_links) >= ADS_LIMIT:
                break
        if ADS_LIMIT and len(all_links) >= ADS_LIMIT:
            break
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    print(f"Total unique detail links collected: {len(all_links)}")
    if ADS_LIMIT:
        all_links = all_links[:ADS_LIMIT]

    # fetch details (itère)
    for i, link in enumerate(all_links, start=1):
        print(f"[{i}/{len(all_links)}] visiting {link}")
        r = safe_get(link)
        if not r:
            print("  failed to fetch detail")
            continue
        try:
            rec = parse_listing_detail(r.text, link)
            scraped.append(rec)
        except Exception as e:
            print("  parse error:", e)
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    # dataframe & export CSV
    df = pd.DataFrame(scraped)
    # colonnes ordonnées demandées
    cols = ["Marque","Modèle","Année","Kilométrage","Carburant","Boîte_vitesse",
            "Puissance_fiscale","Nombre_portes","Etat_generale","Prix","url","title_raw"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    # numeric conversions
    df['Prix'] = pd.to_numeric(df['Prix'], errors='coerce')
    df['Année'] = pd.to_numeric(df['Année'], errors='coerce')
    df['Kilométrage'] = pd.to_numeric(df['Kilométrage'], errors='coerce')
    df['Puissance_fiscale'] = pd.to_numeric(df['Puissance_fiscale'], errors='coerce')
    df['Nombre_portes'] = pd.to_numeric(df['Nombre_portes'], errors='coerce')

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print("Saved", OUTPUT_CSV, "rows:", len(df))

if __name__ == "__main__":
    main()
