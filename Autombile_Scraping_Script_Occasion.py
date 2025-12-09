import time
import re
import json
import csv
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

BASE_URL = "https://www.automobile.tn"
START_URL = "https://www.automobile.tn/fr/occasion?expand=1&brand[]=&model[]="

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
})


def fetch_soup(url: str) -> BeautifulSoup:
    r = session.get(url, timeout=15)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


# ---------- 1. Récupérer toutes les URLs d'occasions ----------

def parse_occasion_urls(soup: BeautifulSoup) -> list[str]:
    urls = []
    for item in soup.select("div.occasion-item-v2"):
        a = item.select_one("a.occasion-link-overlay")
        if not a or not a.get("href"):
            continue
        full_url = urljoin(BASE_URL, a["href"])
        urls.append(full_url)
    return urls


def find_next_page_url(soup: BeautifulSoup, current_url: str) -> str | None:
    a = soup.select_one("ul.pagination a[rel=next]")
    if a and a.get("href"):
        return urljoin(current_url, a["href"])

    a = soup.select_one("ul.pagination li.next a")
    if a and a.get("href"):
        return urljoin(current_url, a["href"])

    for cand in soup.select("ul.pagination a"):
        text = cand.get_text(strip=True).lower()
        if text.startswith("suiv") and cand.get("href"):
            return urljoin(current_url, cand["href"])

    for cand in soup.select("ul.pagination a"):
        href = cand.get("href") or ""
        if "page=" in href:
            return urljoin(current_url, href)

    return None


def get_all_occasion_urls(start_url: str) -> list[str]:
    visited_pages: set[str] = set()
    all_urls: set[str] = set()
    current_url = start_url

    while current_url and current_url not in visited_pages:
        print(f"Fetching occasion page: {current_url}")
        visited_pages.add(current_url)

        soup = fetch_soup(current_url)
        page_urls = parse_occasion_urls(soup)
        print(f"  Found {len(page_urls)} occasion URLs on this page.")

        new_urls = set(page_urls) - all_urls
        print(f"  New URLs: {len(new_urls)}")
        all_urls.update(new_urls)

        next_url = find_next_page_url(soup, current_url)
        if not next_url:
            print("No next page link found, stopping pagination.")
            break

        current_url = next_url
        time.sleep(1.0)

    return sorted(all_urls)


# ---------- 2. Parser une page d'occasion ----------

def parse_brand_model_from_url(annonce_url: str) -> tuple[str, str]:
    path = urlparse(annonce_url).path  # /fr/occasion/mercedes-benz/classe-c/110337
    parts = [p for p in path.split("/") if p]
    try:
        i = parts.index("occasion")
    except ValueError:
        return "", ""
    if len(parts) <= i + 2:
        return "", ""
    brand_slug = parts[i + 1]
    model_slug = parts[i + 2]
    brand = brand_slug.replace("-", " ").title()
    model = model_slug.replace("-", " ").title()
    return brand, model


def parse_jsonld_car(soup: BeautifulSoup) -> dict:
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        if isinstance(data, dict) and data.get("@type") == "Car":
            return data
    return {}


def parse_text_block_specs(soup: BeautifulSoup) -> dict:
    specs = {}
    for li in soup.select("div.divided-specs li"):
        name_el = li.select_one(".spec-name")
        value_el = li.select_one(".spec-value")
        if not name_el or not value_el:
            continue
        label = name_el.get_text(strip=True)
        value = value_el.get_text(" ", strip=True)
        specs[label] = value
    return specs


def extract_int(text: str) -> int | None:
    m = re.search(r"\d+", text or "")
    return int(m.group(0)) if m else None


def parse_summary_block(soup: BeautifulSoup) -> dict:
    res = {}
    road = soup.select_one("li.road")
    if road:
        txt = road.get_text(" ", strip=True)
        res["km"] = extract_int(txt)

    year = soup.select_one("li.year")
    if year:
        txt = year.get_text(" ", strip=True)
        res["year"] = extract_int(txt)

    price_el = soup.select_one("div.price, span.price, .price span")
    if price_el:
        txt = price_el.get_text(" ", strip=True)
        digits = re.sub(r"[^\d]", "", txt)
        if digits:
            res["price"] = int(digits)

    return res


def parse_occasion_row(annonce_url: str) -> dict:
    print(f"Parsing occasion: {annonce_url}")
    soup = fetch_soup(annonce_url)

    marque, modele = parse_brand_model_from_url(annonce_url)
    car_ld = parse_jsonld_car(soup)
    specs = parse_text_block_specs(soup)
    summary = parse_summary_block(soup)

    # Année
    year = None
    if "year" in summary:
        year = summary["year"]
    elif "vehicleModelDate" in car_ld:
        year = extract_int(str(car_ld["vehicleModelDate"]))

    # Kilométrage
    km = None
    if "km" in summary:
        km = summary["km"]
    elif "mileageFromOdometer" in car_ld:
        val = car_ld["mileageFromOdometer"].get("value")
        try:
            km = int(val)
        except Exception:
            km = extract_int(str(val))

    # Carburant
    carburant = ""
    if "Énergie" in specs:
        carburant = specs["Énergie"]
    elif "Energie" in specs:
        carburant = specs["Energie"]
    elif "fuelType" in car_ld:
        carburant = car_ld["fuelType"]

    # Boîte
    boite = specs.get("Boite vitesse", "")

    # Puissance fiscale
    pf = None
    if "Puissance fiscale" in specs:
        pf = extract_int(specs["Puissance fiscale"])

    # Nombre de portes (si dispo dans d'autres annonces)
    nb_portes = None
    if "Nombre de portes" in specs:
        nb_portes = extract_int(specs["Nombre de portes"])

    # Prix
    price = None
    if "price" in summary:
        price = summary["price"]
    elif "offers" in car_ld and isinstance(car_ld["offers"], dict):
        price_val = car_ld["offers"].get("price")
        try:
            price = int(price_val)
        except Exception:
            price = extract_int(str(price_val))

    return {
        "Marque": marque,
        "Modèle": modele,
        "Année": year,
        "Kilométrage": km,
        "Carburant": carburant,
        "Boîte_vitesse": boite,
        "Puissance_fiscale": pf,
        "Nombre_portes": nb_portes,
        "Etat_generale": "occasion",
        "Prix": price,
    }


# ---------- 3. Pipeline complet + CSV ----------

def main():
    # 1) Récupérer toutes les URLs d'annonces
    urls = get_all_occasion_urls(START_URL)
    print(f"\nTotal unique occasion URLs collected: {len(urls)}\n")

    # 2) Parser chaque annonce
    rows: list[dict] = []
    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] {url}")
        try:
            row = parse_occasion_row(url)
            rows.append(row)
        except Exception as e:
            print(f"  Error on {url}: {e}")
        time.sleep(0.5)

    # 3) Sauvegarder en CSV
    fieldnames = [
        "Marque",
        "Modèle",
        "Année",
        "Kilométrage",
        "Carburant",
        "Boîte_vitesse",
        "Puissance_fiscale",
        "Nombre_portes",
        "Etat_generale",
        "Prix",
    ]

    with open("automobile_tn_occasion.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved {len(rows)} rows to automobile_tn_occasion.csv")


if __name__ == "__main__":
    main()
