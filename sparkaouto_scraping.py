#!/usr/bin/env python3
# sparkauto_full_scraper.py
"""
Scraper SparkAuto (https://www.sparkauto.tn)
- Essayez d'obtenir le maximum de données depuis les listings et pages annonces.
- Respecte robots.txt (vérification simple).
- Sauvegarde CSV + JSON.
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import random
import pandas as pd
import urllib.robotparser
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from datetime import datetime
import json

BASE = "https://www.sparkauto.tn"
START_LISTING = BASE + "/"   # la home contient des listings; on peut aussi cibler /occasion ou /voitures si dispo
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SparkAutoScraper/1.0; +https://example.org/bot)"}

# ---------- Settings ----------
MAX_PAGES = 10             # nombre de pages listing à parcourir (adapter)
ADS_PER_PAGE_LIMIT = None  # None => toutes les annonces du listing page ; sinon <= int
FETCH_AD_DETAILS = True    # True => visite chaque annonce pour extraire plus de champs
DELAY_MIN, DELAY_MAX = 0.7, 1.4
TIMEOUT = 12

# ---------- Robots.txt check ----------
def allowed_by_robots(url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # si on ne peut pas lire robots.txt, jouer la prudence mais autoriser en ralentissant
        return True

# ---------- Helpers ----------
def safe_get(url, session=None, timeout=TIMEOUT):
    session = session or requests.Session()
    try:
        r = session.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        print("Request error:", e, "->", url)
        return None

def clean_text(t):
    if not t:
        return None
    return " ".join(t.split()).strip()

def parse_price(text):
    if not text:
        return None
    # retire espaces et prend le nombre avant DT/TND
    m = re.search(r'([0-9]{1,3}(?:[ \.,][0-9]{3})*(?:[ \.,][0-9]+)?)\s*(dt|tnd|tun|dinar)?', text, flags=re.I)
    if m:
        num = m.group(1)
        num = re.sub(r'[^\d]', '', num)
        try:
            return int(num)
        except:
            return clean_text(m.group(0))
    return clean_text(text)

def parse_km(text):
    if not text:
        return None
    m = re.search(r'([0-9][0-9\s\.,]*)\s*(km|kilom)', text, flags=re.I)
    if m:
        return int(re.sub(r'[^\d]', '', m.group(1)))
    return None

def find_nearest_link(elem):
    # cherche un <a href> dans le block, sinon dans parents
    a = elem.find('a', href=True)
    if a:
        return urljoin(BASE, a['href'])
    # fallback: try next sibling anchors
    parent = elem
    for _ in range(3):
        if not parent:
            break
        a = parent.find_next('a', href=True)
        if a:
            return urljoin(BASE, a['href'])
        parent = parent.parent
    return None

# ---------- Listing parsing heuristics ----------
def parse_listing_page(html):
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # strategy:
    # 1) try to find card-like containers (article, .card, .product, .vehicle-card)
    candidates = []
    for sel in ['article', '.card', '.product', '.vehicle-card', '.listing-item', '.car-card', '.vehicle']:
        candidates.extend(soup.select(sel))

    # If no structured cards found, fallback to find headings (h1..h4) which look like titles
    if not candidates:
        headings = soup.find_all(re.compile('^h[1-4]$'))
        for h in headings:
            # simple heuristic: title followed by price or "DT" => treat as ad block
            nxt = "".join([s for s in h.find_all_next(string=True, limit=12)])
            if 'dt' in nxt.lower() or re.search(r'\bkm\b', nxt.lower()):
                candidates.append(h.parent or h)

    seen = set()
    for block in candidates:
        text = block.get_text(" ", strip=True)
        # cheap filter: block must contain money or km
        if not re.search(r'\b(dt|tnd|tun|dinar)\b', text, flags=re.I) and not re.search(r'\bkm\b', text, flags=re.I):
            continue

        # avoid duplicates via text snippet
        key = text[:120]
        if key in seen:
            continue
        seen.add(key)

        # title: look for heading inside block
        title_el = block.find(re.compile('^h[1-4]$'))
        title = clean_text(title_el.get_text(" ", strip=True)) if title_el else None

        # price: find element containing DT or first number with dt
        price = None
        for sel in ['.price', '.prix', '.product-price', '.card-price']:
            p = block.select_one(sel)
            if p and p.get_text(strip=True):
                price = parse_price(p.get_text(" ", strip=True))
                break
        if not price:
            # fallback: search text for DT
            m = re.search(r'([0-9\.\s,]+)\s*(DT|TND|dinar)', text, flags=re.I)
            if m:
                price = parse_price(m.group(0))

        # ref: 'Réf' or 'Ref' near the block
        ref = None
        mref = re.search(r'\b[Rr]éf[:\s]*([0-9\-]+)', text)
        if not mref:
            mref = re.search(r'\b[Rr]ef[:\s]*([0-9\-]+)', text)
        if mref:
            ref = mref.group(1)

        # year
        year = None
        my = re.search(r'\b(19|20)\d{2}\b', text)
        if my:
            year = my.group(0)

        # mileage
        km = parse_km(text)

        # fuel (essence,diesel,hybride,électrique)
        fuel = None
        for f in ['essence', 'diesel', 'hybride', 'electrique', 'electrique', 'gpl']:
            if re.search(r'\b' + re.escape(f) + r'\b', text, flags=re.I):
                fuel = f
                break

        # transmission
        transmission = None
        if re.search(r'\b(automatique|manuelle|manuel|auto)\b', text, flags=re.I):
            mtr = re.search(r'\b(automatique|manuelle|manuel|auto)\b', text, flags=re.I)
            transmission = mtr.group(1) if mtr else None

        # link
        link = find_nearest_link(block)

        # thumbnail or image
        img = None
        img_el = block.find('img')
        if img_el and img_el.get('src'):
            img = urljoin(BASE, img_el.get('src'))

        results.append({
            "title": title,
            "price_raw": price,
            "ref": ref,
            "year": year,
            "mileage_km": km,
            "fuel": fuel,
            "transmission": transmission,
            "url": link,
            "thumb": img,
            "block_text": text[:400]
        })
    return results

# ---------- Detail page parsing ----------
def parse_ad_detail(html, url):
    soup = BeautifulSoup(html, "html.parser")
    # basic fields
    title = None
    try:
        title_el = soup.select_one('h1') or soup.select_one('[data-testid="ad-title"]') or soup.select_one('.product-title')
        title = clean_text(title_el.get_text(" ", strip=True)) if title_el else None
    except:
        title = None

    price = None
    # selectors likely to contain price
    for sel in ['[data-testid="ad-price"]', '.price', '.prix', '.product-price']:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            price = parse_price(el.get_text(" ", strip=True))
            break
    if not price:
        # fallback
        m = re.search(r'([0-9\.\s,]+)\s*(DT|TND|dinar)', soup.get_text(" ", strip=True), flags=re.I)
        if m:
            price = parse_price(m.group(0))

    # Extract specs table / key-value lists
    specs = {}
    # Try common containers: specification lists, property lists
    candidate_containers = soup.select('[data-testid="ad-properties"], .ad-properties, .product-specs, .specs, .car-specs, .attributes') or []
    if candidate_containers:
        for container in candidate_containers:
            # rows might be li, tr, div with label + value
            rows = container.find_all(['li','tr','div'])
            for r in rows:
                txt = r.get_text(" ", strip=True)
                # split by ":" or by strong/label tags
                if ':' in txt:
                    k,v = [p.strip() for p in txt.split(':',1)]
                else:
                    # check for child structure
                    labels = r.find_all(['b','strong','span'])
                    if len(labels) >= 2:
                        k = labels[0].get_text(" ", strip=True)
                        v = labels[-1].get_text(" ", strip=True)
                    else:
                        continue
                if not k: continue
                specs[k] = v

    # as fallback, parse textual heuristics:
    page_text = soup.get_text(" ", strip=True)

    # normalize keys we want:
    def get_field(keys):
        for key in keys:
            for k,v in specs.items():
                if key.lower() in k.lower():
                    return v
        return None

    marque = get_field(['Marque','Brand','marque']) or None
    modele = get_field(['Modèle','Model','modèle']) or None
    annee = get_field(['Année','Année de mise en circulation','year']) or None
    kilometrage = get_field(['Kilométrage','Km','Mileage']) or None
    carburant = get_field(['Carburant','Fuel','Énergie']) or None
    boite = get_field(['Boîte','Transmission','Boite']) or None
    puissance = get_field(['Puissance','Chevaux','CV']) or None
    portes = get_field(['Portes','Nombre de portes']) or None
    couleur = get_field(['Couleur']) or None

    # try regex fallback
    if not annee:
        m = re.search(r'\b(19|20)\d{2}\b', page_text)
        annee = annee or (m.group(0) if m else None)
    if not kilometrage:
        m = re.search(r'([0-9][0-9\s\.,]*)\s*(km|kilom)', page_text, flags=re.I)
        kilometrage = kilometrage or (re.sub(r'[^\d]','',m.group(1)) + " km" if m else None)
    if not carburant:
        for f in ['Essence','Diesel','Hybride','Électrique','GPL','Gaz']:
            if re.search(r'\b'+f+r'\b', page_text, flags=re.I):
                carburant = carburant or f
                break

    # images
    images = []
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src')
        if src and src.strip():
            src = urljoin(BASE, src)
            images.append(src)

    # seller info / location
    seller = None
    location = None
    # try some selectors
    seller_sel = soup.select_one('.seller, .agent, [data-testid="seller"], .dealer')
    if seller_sel:
        seller = clean_text(seller_sel.get_text(" ", strip=True))
    loc_sel = soup.select_one('.location, .city, .ad-location')
    if loc_sel:
        location = clean_text(loc_sel.get_text(" ", strip=True))

    # description
    desc = None
    desc_sel = soup.select_one('.description, [data-testid="ad-description"], .product-desc')
    if desc_sel:
        desc = clean_text(desc_sel.get_text(" ", strip=True))
    else:
        # fallback: look for a long paragraph
        p = soup.find('p')
        if p and len(p.get_text(" ",strip=True)) > 40:
            desc = clean_text(p.get_text(" ", strip=True))

    return {
        "title": title,
        "price": price,
        "marque": marque,
        "modele": modele,
        "annee": annee,
        "kilometrage": kilometrage,
        "carburant": carburant,
        "boite": boite,
        "puissance": puissance,
        "portes": portes,
        "couleur": couleur,
        "images": images,
        "seller": seller,
        "location": location,
        "description": desc,
        "raw_specs": specs
    }

# ---------- Main flow ----------
def collect_listing_urls(start_url=START_LISTING, max_pages=MAX_PAGES, ads_per_page_limit=ADS_PER_PAGE_LIMIT):
    s = requests.Session()
    urls = []
    page_url = start_url
    for p in range(1, max_pages+1):
        # try param-based pagination: guess ?page=n or /page/n
        # first try homepage for p==1
        if p == 1:
            url = start_url
        else:
            # try "?page="
            url = start_url
            if '?' in url:
                url = url + "&page=" + str(p)
            else:
                url = url.rstrip('/') + "?page=" + str(p)

        print("Fetching listing page:", url)
        if not allowed_by_robots(url, HEADERS['User-Agent']):
            print("Robots.txt disallow for", url, "- skip")
            break

        r = safe_get(url, session=s)
        if not r:
            break

        items = parse_listing_page(r.text)
        if not items:
            print("No items found on listing page (structure may differ). Stopping pagination.")
            break

        # collect links for detail fetch
        for it in items:
            if it.get("url"):
                urls.append(it)
            else:
                # still store block info, might parse detail later if link missing
                urls.append(it)
            if ads_per_page_limit and len(urls) >= ads_per_page_limit:
                break

        # politeness
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    # deduplicate by URL or ref+title
    uniq = []
    seen = set()
    for it in urls:
        key = it.get('url') or (it.get('ref') or "") + (it.get('title') or "")
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    return uniq

def main():
    print("SparkAuto full scraper start:", datetime.utcnow().isoformat())
    start_time = time.time()
    s = requests.Session()

    # collect listing urls
    items = collect_listing_urls(start_url=BASE + "/", max_pages=MAX_PAGES, ads_per_page_limit=None)
    print("Listing items collected:", len(items))

    results = []
    for idx, it in enumerate(tqdm(items, desc="Scraping ads")):
        base_record = {
            "scrape_date": datetime.utcnow().isoformat(),
            "title_listing": it.get("title"),
            "price_listing": it.get("price_raw"),
            "ref": it.get("ref"),
            "year_listing": it.get("year"),
            "mileage_listing": it.get("mileage_km"),
            "fuel_listing": it.get("fuel"),
            "transmission_listing": it.get("transmission"),
            "url": it.get("url"),
            "thumb": it.get("thumb")
        }

        if FETCH_AD_DETAILS and it.get("url"):
            # check robots
            if not allowed_by_robots(it['url'], HEADERS['User-Agent']):
                print("Robots disallow detail:", it['url'])
                detail = {}
            else:
                r = safe_get(it['url'], session=s)
                if not r:
                    detail = {}
                else:
                    detail = parse_ad_detail(r.text, it['url'])
            # merge
            base_record.update(detail)
            # add raw listing block if desired
            base_record['block_text'] = it.get('block_text')
        else:
            base_record.update({
                "marque": None,
                "modele": None,
                "annee": it.get("year"),
                "kilometrage": it.get("mileage_km"),
            })

        results.append(base_record)

        # politeness & stop if too many
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        # optional short-circuit: remove to scrape all
        # if idx>200: break

    # save CSV + JSON
    df = pd.DataFrame(results)
    csv_name = f"sparkauto_scrape_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_name, index=False, encoding="utf-8-sig")
    with open(csv_name.replace('.csv', '.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved:", csv_name, "and JSON.")
    print("Time elapsed:", time.time() - start_time)

if __name__ == "__main__":
    main()
