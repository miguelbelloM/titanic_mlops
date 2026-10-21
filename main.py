# API FastAPI que recibe los datos de un estudiante y consulta el endpoint MLOPS_EAN de Databricks
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Credenciales: se inyectan como variables de entorno en Azure (nunca escritas en el código)
DATABRICKS_URL = os.getenv("DATABRICKS_URL")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

app = FastAPI(title="API MLOps EAN", description="Predicción de aprobación de estudiantes")

# Permite que el frontend (index.html) llame a la API desde otro dominio
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Campos que espera el modelo (deben coincidir con la firma registrada en MLflow)
class Estudiante(BaseModel):
    horas_estudio: float = Field(..., ge=0, le=24)
    horas_sueno: float = Field(..., ge=0, le=24)
    asistencia: float = Field(..., ge=0, le=100)
    promedio_previo: float = Field(..., ge=0, le=5)
    indice_dedicacion: float = Field(..., ge=0)
    en_riesgo: int = Field(..., ge=0, le=1)
    destacado: int = Field(..., ge=0, le=1)


@app.get("/health")
def health():
    # Verifica que la API está viva y que tiene configuradas las credenciales
    return {"status": "ok", "databricks_configurado": bool(DATABRICKS_URL and DATABRICKS_TOKEN)}


@app.post("/predecir")
def predecir(est: Estudiante):
    if not (DATABRICKS_URL and DATABRICKS_TOKEN):
        raise HTTPException(status_code=500, detail="Faltan DATABRICKS_URL o DATABRICKS_TOKEN")

    # Databricks exige el formato dataframe_records
    payload = {"dataframe_records": [est.model_dump()]}

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

    pred = int(r.json()["predictions"][0])  # 1 = aprueba, 0 = reprueba
    return {
        "prediccion": pred,
        "resultado": "Aprueba ✅" if pred == 1 else "Reprueba ❌",
        "datos_recibidos": est.model_dump(),
    }
