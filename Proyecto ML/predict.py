import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf  # pyright: ignore[reportMissingModuleSource]
from tensorflow import keras  # pyright: ignore[reportMissingModuleSource]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_processed"
MODEL_DIR = BASE_DIR / "saved_models"

# Columnas que debe traer un CSV nuevo (el mismo formato del test.csv de Kaggle).
# "Cabin" es opcional: si no viene, se asume que ningun pasajero tiene cabina registrada.
COLUMNAS_OBLIGATORIAS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def cargar_modelo():
    """Carga el mejor modelo elegido por tabla_comparativa.py."""
    with open(DATA_DIR / "mejor_modelo.json", encoding="utf-8") as f:
        info = json.load(f)
    modelo = keras.models.load_model(MODEL_DIR / info["archivo"])
    return modelo, info


def preprocesar_csv(df):
    """Aplica EXACTAMENTE el mismo preprocesamiento del entrenamiento a un DataFrame nuevo.
    Usa los valores guardados en preprocessing_params.json (calculados solo con train)."""
    with open(DATA_DIR / "preprocessing_params.json", encoding="utf-8") as f:
        p = json.load(f)

    faltan = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan estas columnas: {faltan}. "
                         f"Debe tener: {COLUMNAS_OBLIGATORIAS} (y opcionalmente Cabin).")

    out = df.copy()
    if "Cabin" not in out.columns:
        out["Cabin"] = np.nan

    # Ingenieria de caracteristicas (igual que en eda_preprocessing.py)
    title = out["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    out["Title"] = title.where(title.isin(p["titulos_validos"]), "Rare")
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    out = out[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
               "Title", "FamilySize", "IsAlone", "HasCabin"]]

    # Imputacion con los valores de train guardados
    out["Age"] = out["Age"].fillna(p["age_median"])
    out["Fare"] = out["Fare"].fillna(p["fare_median"])
    out["Embarked"] = out["Embarked"].fillna(p["embarked_mode"])
    if p["fare_log1p"]:
        out["Fare"] = np.log1p(out["Fare"])

    # One-Hot con las MISMAS columnas y en el MISMO orden que en el entrenamiento
    enc = pd.get_dummies(out, columns=p["cat_cols"], dtype=float)
    enc = enc.reindex(columns=p["columnas_finales"], fill_value=0.0)

    # Normalizacion con la media y varianza de train
    num = p["num_cols"]
    media = np.array(p["norm_mean"], dtype="float32")
    desv = np.maximum(np.sqrt(np.array(p["norm_variance"], dtype="float32")), 1e-7)
    enc[num] = (enc[num].values.astype("float32") - media) / desv
    return enc.values.astype("float32")


def predecir_csv(ruta_csv, umbral=0.5):
    """Lee un CSV nuevo, lo preprocesa y devuelve un DataFrame con las predicciones."""
    df = pd.read_csv(ruta_csv)
    X = preprocesar_csv(df)
    modelo, info = cargar_modelo()
    prob = modelo.predict(X, verbose=0).ravel()
    resultado = pd.DataFrame({
        "PassengerId": df["PassengerId"] if "PassengerId" in df.columns else np.arange(1, len(df) + 1),
        "Probabilidad_supervivencia": prob.round(4),
        "Survived": (prob >= umbral).astype(int),
    })
    return resultado, df, info


def generar_submission():
    """Genera submission.csv para Kaggle con el test oficial ya preprocesado."""
    modelo, info = cargar_modelo()
    X_test = np.load(DATA_DIR / "X_test.npy").astype("float32")
    ids = np.load(DATA_DIR / "passenger_ids_test.npy")
    prob = modelo.predict(X_test, verbose=0).ravel()
    sub = pd.DataFrame({"PassengerId": ids, "Survived": (prob >= 0.5).astype(int)})
    sub.to_csv(BASE_DIR / "submission.csv", index=False)
    print(f"Modelo usado: {info['titulo']}")
    print(f"submission.csv creado con {len(sub)} filas. Sobrevivientes predichos: {sub['Survived'].sum()}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Sin argumentos: submission para Kaggle
        generar_submission()
    else:
        # Con un CSV: prediccion sobre datos nuevos
        ruta = Path(sys.argv[1])
        resultado, df, info = predecir_csv(ruta)
        salida = BASE_DIR / f"predicciones_{ruta.stem}.csv"
        resultado.to_csv(salida, index=False)
        print(f"Modelo usado: {info['titulo']}")
        print(f"{len(resultado)} pasajeros procesados. Resultado guardado en: {salida.name}")
        print(resultado.head(10).to_string(index=False))
        if "Survived" in df.columns:
            acc = (resultado["Survived"].values == df["Survived"].values).mean()
            print(f"\nEl CSV traia la columna Survived: accuracy = {acc:.4f}")