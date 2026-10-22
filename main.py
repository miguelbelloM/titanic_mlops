# API FastAPI que recibe los datos de un pasajero del Titanic y consulta el endpoint titanic_EAN de Databricks
import os
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Credenciales: se inyectan como variables de entorno en Azure (nunca escritas en el código)
DATABRICKS_URL = os.getenv("DATABRICKS_URL")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Codificación de Sex usada al entrenar.
# AJUSTAR si en el notebook se usó otra (p. ej. LabelEncoder da female=0, male=1).
SEX_MAP = {"male": 0, "female": 1}

app = FastAPI(title="API Titanic EAN", description="Predicción de supervivencia de pasajeros del Titanic")

# Permite que un frontend llame a la API desde otro dominio
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Entrada "amigable": el usuario envía texto y la API lo convierte a lo que espera el modelo
class Pasajero(BaseModel):
    pclass: Literal[1, 2, 3] = Field(..., description="Clase del boleto: 1, 2 o 3")
    sex: Literal["male", "female"] = Field(..., description="Sexo: male o female")
    age: float = Field(..., ge=0, le=100, description="Edad en años")
    sibsp: int = Field(..., ge=0, le=10, description="Hermanos/cónyuges a bordo")
    parch: int = Field(..., ge=0, le=10, description="Padres/hijos a bordo")
    fare: float = Field(..., ge=0, description="Precio del boleto")
    embarked: Literal["C", "Q", "S"] = Field(..., description="Puerto: C=Cherbourg, Q=Queenstown, S=Southampton")


def a_features(p: Pasajero) -> dict:
    # Mismo formato que la firma registrada en MLflow (8 columnas, Embarked en one-hot con C como base)
    return {
        "Pclass": p.pclass,
        "Sex": SEX_MAP[p.sex],
        "Age": float(p.age),
        "SibSp": p.sibsp,
        "Parch": p.parch,
        "Fare": float(p.fare),
        "Embarked_Q": 1 if p.embarked == "Q" else 0,
        "Embarked_S": 1 if p.embarked == "S" else 0,
    }


@app.get("/")
def root():
    return {"mensaje": "API Titanic EAN activa", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    # Verifica que la API está viva y que tiene configuradas las credenciales
    return {"status": "ok", "databricks_configurado": bool(DATABRICKS_URL and DATABRICKS_TOKEN)}


@app.post("/predecir")
def predecir(p: Pasajero):
    if not (DATABRICKS_URL and DATABRICKS_TOKEN):
        raise HTTPException(status_code=500, detail="Faltan DATABRICKS_URL o DATABRICKS_TOKEN")

    features = a_features(p)
    payload = {"dataframe_records": [features]}

    try:
        r = requests.post(
            DATABRICKS_URL,
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json=payload,
            timeout=120,  # margen para el arranque en frío del endpoint
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Databricks tardó demasiado (arranque en frío). Intente de nuevo.")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error de Databricks: {r.text}")

    pred = int(r.json()["predictions"][0])  # 1 = sobrevive, 0 = no sobrevive
    return {
        "prediccion": pred,
        "resultado": "Sobrevive ✅" if pred == 1 else "No sobrevive ❌",
        "datos_recibidos": p.model_dump(),
        "features_enviadas": features,
    }
