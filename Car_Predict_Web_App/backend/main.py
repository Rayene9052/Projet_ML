from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "cars_dataset_cleaned_more_final.csv"
MODEL_PATH = BASE_DIR / "car_price_catboost_final.cbm"
MARQUE_MODELES_PATH = BASE_DIR / "marque_modeles.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

YEAR_REFERENCE = 2025

# CatBoost feature metadata (extracted from the trained model)
FEATURES: List[str] = [
    "Age",
    "Kilométrage",
    "Puissance_fiscale",
    "Log_Kilometre",
    "Log_Puissance",
    "Usure_km_par_an",
    "Puiss_Par_Age",
    "Puiss_Par_Km",
    "Prix_per_km",
    "Prix_moy_marque",
    "Prix_median_marque",
    "Prix_std_marque",
    "Count_marque",
    "Prix_moy_modele",
    "Prix_median_modele",
    "Count_modele",
    "Prix_moy_marque_fuel",
    "Marque",
    "Modèle",
    "Carburant_simplifié",
    "Boîte_vitesse",
    "Etat_generale",
    "Km_category",
    "Age_category",
]

CAT_COLS: List[str] = [
    "Marque",
    "Modèle",
    "Carburant_simplifié",
    "Boîte_vitesse",
    "Etat_generale",
    "Km_category",
    "Age_category",
]


class PredictRequest(BaseModel):
    marque: str = Field(..., description="Marque du véhicule")
    modele: str = Field(..., description="Modèle du véhicule")
    annee: int = Field(..., description="Année de mise en circulation")
    kilometrage: int = Field(..., ge=0, description="Kilométrage total")
    carburant: str = Field(..., description="Type de carburant")
    boite_vitesse: str = Field(..., description="Type de boîte de vitesse")
    puissance_fiscale: int = Field(..., gt=0, description="Puissance fiscale (CV)")
    etat_generale: str = Field(..., description="État général (1ère main ou Occasion)")

    @validator("marque", "modele", "carburant", "boite_vitesse", "etat_generale")
    def strip_strings(cls, v: str) -> str:
        return v.strip()


class DatasetInfo(BaseModel):
    prix_global_mean: float
    prix_moy_marque: Dict[str, float]
    marque_count: Dict[str, int]
    prix_moy_modele: Dict[str, float]
    modele_count: Dict[str, int]
    prix_moy_marque_fuel: Dict[str, float]
    marques: List[str]
    carburants: List[str]
    boites: List[str]
    etats: List[str]
    marque_modele_map: Dict[str, List[str]]


def simplifier_carburant(value: str) -> str:
    val = (value or "").lower()
    if "elect" in val:
        return "Electrique"
    if "hybride" in val:
        return "Hybride"
    return "Thermique"


def load_marque_modele_map() -> Dict[str, List[str]]:
    if not MARQUE_MODELES_PATH.exists():
        return {}

    df_map = pd.read_csv(MARQUE_MODELES_PATH, encoding="latin1")
    # Corriger d'éventuels encodages cassés sur les noms de colonnes
    rename_map = {}
    for col in df_map.columns:
        fixed = col
        fixed = fixed.replace("Marque", "marque").replace("marque", "marque")
        fixed = fixed.replace("Modele", "modele").replace("Modèle", "modele")
        fixed = fixed.replace("ModÃ¨le", "modele").replace("Mod�le", "modele")
        if fixed != col:
            rename_map[col] = fixed
    if rename_map:
        df_map = df_map.rename(columns=rename_map)

    mapping: Dict[str, List[str]] = {}
    for _, row in df_map.iterrows():
        marque = str(row.get("marque", "")).strip()
        raw_models = str(row.get("modele", "")).split(",")
        models = [m.strip() for m in raw_models if m.strip()]
        if marque:
            mapping[marque] = models
    return mapping


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige les encodages cassés éventuels sur les noms de colonnes."""
    rename_map = {}
    for col in df.columns:
        fixed = col
        fixed = fixed.replace("ModÃ¨le", "Modèle").replace("Mod�le", "Modèle")
        fixed = fixed.replace("AnnÃ©e", "Année").replace("Ann�e", "Année")
        fixed = fixed.replace("KilomÃ©trage", "Kilométrage").replace("Kilom�trage", "Kilométrage")
        fixed = fixed.replace("BoÃ®te_vitesse", "Boîte_vitesse").replace("Bo�te_vitesse", "Boîte_vitesse")
        fixed = fixed.replace("Ã‰tat_generale", "Etat_generale").replace("�tat_generale", "Etat_generale")
        if fixed != col:
            rename_map[col] = fixed
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_dataset_info() -> DatasetInfo:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable: {DATA_PATH}")

    try:
        df = pd.read_csv(DATA_PATH, encoding="latin1")
    except UnicodeDecodeError:
        # Fallback: réouverture avec remplacement des caractères invalides
        with DATA_PATH.open("r", encoding="latin1", errors="replace") as f:
            df = pd.read_csv(f)

    df = _normalize_columns(df)

    # Nettoyage simple
    try:
        df["Marque"] = df["Marque"].astype(str).str.strip()
        df["Modèle"] = df["Modèle"].astype(str).str.strip()
        df["Carburant"] = df["Carburant"].astype(str).str.strip()
        df["Boîte_vitesse"] = df["Boîte_vitesse"].astype(str).str.strip()
        df["Etat_generale"] = df["Etat_generale"].astype(str).str.strip()
    except KeyError as exc:
        missing = [k for k in ["Marque", "Modèle", "Carburant", "Boîte_vitesse", "Etat_generale"] if k not in df.columns]
        raise KeyError(f"Colonnes manquantes après normalisation: {missing}") from exc

    prix_global_mean = float(df["Prix"].mean())

    prix_moy_marque = df.groupby("Marque")["Prix"].mean().to_dict()
    marque_count = df.groupby("Marque")["Prix"].size().to_dict()

    prix_moy_modele = df.groupby("Modèle")["Prix"].mean().to_dict()
    modele_count = df.groupby("Modèle")["Prix"].size().to_dict()

    prix_moy_marque_fuel = (
        df.groupby(["Marque", "Carburant"])["Prix"].mean().to_dict()
    )
    prix_moy_marque_fuel = {
        json.dumps({"marque": k[0], "carburant": k[1]}): float(v)
        for k, v in prix_moy_marque_fuel.items()
    }

    options_marques = sorted(df["Marque"].unique().tolist())
    options_carburants = sorted(df["Carburant"].unique().tolist())
    options_boites = sorted(df["Boîte_vitesse"].unique().tolist())
    options_etats = sorted(df["Etat_generale"].unique().tolist())

    return DatasetInfo(
        prix_global_mean=prix_global_mean,
        prix_moy_marque={k: float(v) for k, v in prix_moy_marque.items()},
        marque_count={k: int(v) for k, v in marque_count.items()},
        prix_moy_modele={k: float(v) for k, v in prix_moy_modele.items()},
        modele_count={k: int(v) for k, v in modele_count.items()},
        prix_moy_marque_fuel=prix_moy_marque_fuel,
        marques=options_marques,
        carburants=options_carburants,
        boites=options_boites,
        etats=options_etats,
        marque_modele_map=load_marque_modele_map(),
    )


def format_marque_modele_key(marque: str, carburant: str) -> str:
    return json.dumps({"marque": marque, "carburant": carburant})


def build_features(payload: PredictRequest, dataset_info: DatasetInfo) -> pd.DataFrame:
    df_input = pd.DataFrame(
        {
            "Marque": [payload.marque],
            "Modèle": [payload.modele],
            "Année": [payload.annee],
            "Kilométrage": [payload.kilometrage],
            "Carburant": [payload.carburant],
            "Boîte_vitesse": [payload.boite_vitesse],
            "Puissance_fiscale": [payload.puissance_fiscale],
            "Etat_generale": [payload.etat_generale],
        }
    )

    df_input["Age"] = YEAR_REFERENCE - df_input["Année"]
    df_input["Age"] = df_input["Age"].clip(lower=0, upper=50)

    df_input["Usure_km_par_an"] = df_input["Kilométrage"] / df_input["Age"].replace(0, 1)
    df_input["Usure_km_par_an"] = df_input["Usure_km_par_an"].clip(upper=50000)

    df_input["Log_Kilometre"] = np.log1p(df_input["Kilométrage"])
    df_input["Log_Puissance"] = np.log1p(df_input["Puissance_fiscale"])

    marque = df_input.at[0, "Marque"]
    modele = df_input.at[0, "Modèle"]
    carburant = df_input.at[0, "Carburant"]

    df_input["Prix_moy_marque"] = dataset_info.prix_moy_marque.get(
        marque, dataset_info.prix_global_mean
    )
    df_input["Prix_median_marque"] = df_input["Prix_moy_marque"]
    df_input["Prix_std_marque"] = 0
    df_input["Count_marque"] = dataset_info.marque_count.get(marque, 1)

    df_input["Prix_moy_modele"] = dataset_info.prix_moy_modele.get(
        modele, dataset_info.prix_global_mean
    )
    df_input["Prix_median_modele"] = df_input["Prix_moy_modele"]
    df_input["Count_modele"] = dataset_info.modele_count.get(modele, 1)

    key_fuel = format_marque_modele_key(marque, carburant)
    df_input["Prix_moy_marque_fuel"] = dataset_info.prix_moy_marque_fuel.get(
        key_fuel, dataset_info.prix_global_mean
    )

    df_input["Puiss_Par_Age"] = df_input["Puissance_fiscale"] / df_input["Age"].replace(0, 1)
    df_input["Puiss_Par_Km"] = df_input["Puissance_fiscale"] / (df_input["Kilométrage"] + 1)
    df_input["Prix_per_km"] = dataset_info.prix_global_mean / (df_input["Kilométrage"] + 1)

    df_input["Carburant_simplifié"] = df_input["Carburant"].apply(simplifier_carburant)

    df_input["Km_category"] = pd.cut(
        df_input["Kilométrage"],
        bins=[0, 50000, 100000, 150000, 250000, float("inf")],
        labels=["Très_faible", "Faible", "Moyen", "Élevé", "Très_élevé"],
    ).astype(str)

    df_input["Age_category"] = pd.cut(
        df_input["Age"],
        bins=[0, 3, 7, 12, 20, 100],
        labels=["Neuf", "Récent", "Moyen_age", "Ancien", "Très_ancien"],
    ).astype(str)

    for col in CAT_COLS:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str)

    return df_input[FEATURES]


def create_app() -> FastAPI:
    app = FastAPI(
        title="API Prédiction Prix Voiture - Tunisie",
        version="1.0.0",
        description="API pour prédire le prix des voitures en Tunisie à partir d'un modèle CatBoost entraîné.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))

    dataset_info = load_dataset_info()

    @app.get("/api/options")
    def get_options():
        return {
            "marques": dataset_info.marques,
            "carburants": dataset_info.carburants,
            "boites": dataset_info.boites,
            "etats": dataset_info.etats,
        }

    @app.get("/api/modeles")
    def get_modeles(
        marque: str = Query(..., description="Marque pour filtrer les modèles")
    ):
        marque_clean = marque.strip()
        modeles = dataset_info.marque_modele_map.get(marque_clean)

        if modeles is None:
            # fallback basé sur le dataset
            try:
                df = pd.read_csv(DATA_PATH, encoding="latin1")
            except UnicodeDecodeError:
                with DATA_PATH.open("r", encoding="latin1", errors="replace") as f:
                    df = pd.read_csv(f)
            df = _normalize_columns(df)
            df = df[df["Marque"] == marque_clean]
            modeles = sorted(df["Modèle"].unique().tolist())

        if not modeles:
            raise HTTPException(status_code=404, detail="Aucun modèle trouvé pour cette marque")

        return {"marque": marque_clean, "modeles": modeles}

    @app.post("/api/predict")
    def predict_price(payload: PredictRequest):
        try:
            features_df = build_features(payload, dataset_info)
            preds_log = model.predict(features_df)
            predicted_price = float(np.expm1(preds_log)[0])
        except Exception as exc:  # pragma: no cover - defensive logging
            raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction: {exc}")

        return {
            "prix_estime": round(predicted_price, 2),
            "fourchette": {
                "min": round(predicted_price * 0.945, 0),
                "max": round(predicted_price * 1.065, 0),
            },
            "devise": "TND",
        }

    @app.get("/health")
    def healthcheck():
        return {"status": "ok"}

    if FRONTEND_DIR.exists():
        app.mount(
            "/app",
            StaticFiles(directory=str(FRONTEND_DIR), html=True),
            name="app",
        )

        @app.get("/")
        def root_redirect():
            return RedirectResponse(url="/app/")

    return app


app = create_app()


