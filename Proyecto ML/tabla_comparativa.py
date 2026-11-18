import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_processed"
GRAPH_DIR = BASE_DIR / "graficas"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Criterio para elegir el mejor modelo (se mide en VALIDACION, nunca en test).
# AUC-ROC es mas estable que Accuracy con pocos datos: con 134 pasajeros,
# un solo pasajero cambia el accuracy en 0.75 puntos.
CRITERIO = "AUC-ROC"

# Titulo del modelo -> nombre del archivo .keras guardado en saved_models/
ARCHIVOS = {
    "Modelo 1 (MLP Base)": "m1_mlp_base",
    "Modelo 2 (L2 Ridge)": "m2_l2_ridge",
    "Modelo 3 (L1 Lasso)": "m3_l1_lasso",
    "Modelo 4 (ElasticNet)": "m4_elasticnet",
    "Modelo 5 (Dropout)": "m5_dropout",
    "Modelo 6 (BatchNorm)": "m6_batchnorm",
    "Modelo 7 (Autoencoder + clasificador)": "m7_autoencoder",
    "Modelo 7 (Autoencoder)": "m7_autoencoder",
    "Modelo 8 (Conv1D)": "m8_conv1d",
}
# 1. Cargar resultados de los dos grupos
results = {}
for nombre_json in ["results_grupo1.json", "results_grupo2.json"]:
    ruta = DATA_DIR / nombre_json
    if ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            results.update(json.load(f))
    else:
        print(f"AVISO: falta {nombre_json}. Ejecuta primero el script de ese grupo.")

if not results:
    raise SystemExit("No hay resultados que comparar.")

# 2. Construir la tabla comparativa
METRICAS = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Loss"]
filas = []
for titulo, r in results.items():
    fila = {"Modelo": titulo}
    for m in METRICAS:
        fila[f"Val {m}"] = r["Validacion"][m]
    if "Test" in r:
        for m in METRICAS:
            fila[f"Test {m}"] = r["Test"][m]
    fila["Epocas"] = r["Epocas ejecutadas"]
    filas.append(fila)

tabla = pd.DataFrame(filas)
tabla = tabla.sort_values("Modelo").reset_index(drop=True)
tabla.to_csv(DATA_DIR / "tabla_comparativa.csv", index=False, encoding="utf-8-sig")

# 3. Elegir el mejor modelo por VALIDACION (desempate: accuracy de validacion)
col = f"Val {CRITERIO}"
orden = tabla.sort_values([col, "Val Accuracy"], ascending=False).reset_index(drop=True)
mejor = orden.iloc[0]
mejor_titulo = mejor["Modelo"]
mejor_archivo = ARCHIVOS.get(mejor_titulo)

# Linea base: predecir siempre la clase mayoritaria
y_val = np.load(DATA_DIR / "y_val.npy")
base_val = max(y_val.mean(), 1 - y_val.mean())

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n=== TABLA COMPARATIVA (validacion y test propio) ===")
print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nLinea base (predecir siempre la clase mayoritaria): accuracy = {base_val:.4f}")
print(f"\nMEJOR MODELO segun {col}: {mejor_titulo}  ({col} = {mejor[col]:.4f})")
if "Test AUC-ROC" in tabla.columns:
    print(f"Su rendimiento en el TEST PROPIO: Accuracy = {mejor['Test Accuracy']:.4f} | "
          f"F1 = {mejor['Test F1-Score']:.4f} | AUC = {mejor['Test AUC-ROC']:.4f}")

with open(DATA_DIR / "mejor_modelo.json", "w", encoding="utf-8") as f:
    json.dump({
        "titulo": mejor_titulo,
        "archivo": f"{mejor_archivo}.keras",
        "criterio": CRITERIO,
        "valor_validacion": float(mejor[col]),
    }, f, indent=4)

# 4. Grafica: barras de Accuracy y AUC (validacion vs test)
corto = [t.split("(")[0].strip() + "\n" + t.split("(")[1].split(")")[0][:12] for t in tabla["Modelo"]]
x = np.arange(len(tabla))
tiene_test = "Test AUC-ROC" in tabla.columns
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, m in zip(axes, ["Accuracy", "AUC-ROC"]):
    ancho = 0.38 if tiene_test else 0.6
    ax.bar(x - (ancho / 2 if tiene_test else 0), tabla[f"Val {m}"], ancho, label="Validación")
    if tiene_test:
        ax.bar(x + ancho / 2, tabla[f"Test {m}"], ancho, label="Test propio")
    if m == "Accuracy":
        ax.axhline(base_val, color="red", linestyle="--", label="Línea base")
    ax.set_xticks(x)
    ax.set_xticklabels(corto, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"{m} por modelo")
    ax.legend()
plt.tight_layout()
plt.savefig(GRAPH_DIR / "comparativa_modelos.png", dpi=300)
plt.close()

# 5. Grafica: la tabla comparativa como imagen (para el informe)
cols_img = ["Modelo", "Val Accuracy", "Val Precision", "Val Recall", "Val F1-Score", "Val AUC-ROC"]
if tiene_test:
    cols_img += ["Test Accuracy", "Test F1-Score", "Test AUC-ROC"]
datos = tabla[cols_img].copy()
datos["Modelo"] = datos["Modelo"].str.replace(" + clasificador", "", regex=False)
for c in cols_img[1:]:
    datos[c] = datos[c].map(lambda v: f"{v:.3f}")
fig, ax = plt.subplots(figsize=(15, 0.45 * len(datos) + 1.2))
ax.axis("off")
t = ax.table(cellText=datos.values, colLabels=cols_img, loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(8)
t.scale(1, 1.6)
for j in range(len(cols_img)):
    t[(0, j)].set_facecolor("#1F4E78")
    t[(0, j)].set_text_props(color="white", weight="bold")
idx_mejor = tabla.index[tabla["Modelo"] == mejor_titulo][0] + 1
for j in range(len(cols_img)):
    t[(idx_mejor, j)].set_facecolor("#C6EFCE")
plt.title(f"Tabla comparativa (en verde: mejor modelo según {col})")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "tabla_comparativa.png", dpi=300)
plt.close()

# 6. Grafica: matrices de confusion de los 8 modelos (test propio si existe)
conjunto = "Test" if tiene_test else "Validacion"
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, (titulo, r) in zip(axes.ravel(), sorted(results.items())):
    cm = np.array(r[conjunto]["Matriz de confusion"])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Sí"])
    ax.set_yticklabels(["No", "Sí"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo.split("(")[0].strip() + "\n" + titulo.split("(")[1].split(")")[0][:20], fontsize=9)
plt.suptitle(f"Matrices de confusión ({conjunto})")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "matrices_confusion.png", dpi=300)
plt.close()

print("\nGuardado: tabla_comparativa.csv, mejor_modelo.json y 3 imagenes en /graficas")