import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # pyright: ignore[reportMissingModuleSource]
from tensorflow import keras  # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras import layers, regularizers, callbacks  # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_processed"
MODEL_DIR = BASE_DIR / "saved_models"
GRAPH_DIR = BASE_DIR / "graficas"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos procesados por eda_preprocessing.py
X_train = np.load(DATA_DIR / "X_train.npy").astype("float32")
X_val = np.load(DATA_DIR / "X_val.npy").astype("float32")
y_train = np.load(DATA_DIR / "y_train.npy").astype("float32")
y_val = np.load(DATA_DIR / "y_val.npy").astype("float32")

# Test propio con etiquetas (lo debe generar P1; es distinto del test.csv de Kaggle)
HAY_TEST = (DATA_DIR / "X_holdout.npy").exists() and (DATA_DIR / "y_holdout.npy").exists()
if HAY_TEST:
    X_test = np.load(DATA_DIR / "X_holdout.npy").astype("float32")
    y_test = np.load(DATA_DIR / "y_holdout.npy").astype("float32")
else:
    print("AVISO: no hay X_holdout.npy / y_holdout.npy. Solo se evaluara en validacion.")

n_features = X_train.shape[1]
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Features: {n_features}")
print(f"Proporcion de supervivientes en train: {y_train.mean():.3f}")


# 2. Pipeline tf.data
def make_ds(X, y, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Callbacks obligatorios de Keras
def get_callbacks(name, guardar=True):
    cbs = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
    ]
    if guardar:
        cbs.append(callbacks.ModelCheckpoint(MODEL_DIR / f"{name}.keras",
                                             monitor='val_loss', save_best_only=True))
    return cbs


# Curvas de aprendizaje (Loss y Accuracy)
def plot_history(hist, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(hist.history['loss'], label='Entrenamiento')
    ax1.plot(hist.history['val_loss'], label='Validación')
    ax1.set_title(f'Pérdida (Loss) - {name}')
    ax1.set_xlabel('Época')
    ax1.legend()

    ax2.plot(hist.history['accuracy'], label='Entrenamiento')
    ax2.plot(hist.history['val_accuracy'], label='Validación')
    ax2.set_title(f'Exactitud (Accuracy) - {name}')
    ax2.set_xlabel('Época')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f"curvas_{name}.png", dpi=300)
    plt.close()


# 3. Construccion de modelos
def crear_reg(tipo, l1=0.0, l2=0.0):
    if tipo == "l2":
        return regularizers.l2(l2)
    if tipo == "l1":
        return regularizers.l1(l1)
    if tipo == "l1_l2":
        return regularizers.l1_l2(l1=l1, l2=l2)
    return None


def build_model(tipo=None, l1=0.0, l2=0.0, lr=0.001):
    # Se crea un regularizador por capa
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation='relu', kernel_regularizer=crear_reg(tipo, l1, l2)),
        layers.Dense(32, activation='relu', kernel_regularizer=crear_reg(tipo, l1, l2)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model


# 4. Evaluacion completa
def evaluate(model, X, y):
    probs = model.predict(make_ds(X, y, 128), verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    # Loss de clasificacion pura (sin el termino de regularizacion)
    loss = float(np.mean(keras.losses.binary_crossentropy(y.reshape(-1, 1),
                                                          probs.reshape(-1, 1))))
    return {
        "Accuracy": float(accuracy_score(y, preds)),
        "Precision": float(precision_score(y, preds, zero_division=0)),
        "Recall": float(recall_score(y, preds, zero_division=0)),
        "F1-Score": float(f1_score(y, preds, zero_division=0)),
        "AUC-ROC": float(roc_auc_score(y, probs)),
        "Loss": loss,
        "Matriz de confusion": confusion_matrix(y, preds).tolist(),
    }


def correr(nombre, tipo, l1, l2, lr, batch, epochs=80, base=False):
    keras.utils.set_random_seed(SEED)  # misma semilla para comparar justo
    model = build_model(tipo, l1, l2, lr)
    hist = model.fit(make_ds(X_train, y_train, batch, shuffle=True),
                     validation_data=make_ds(X_val, y_val, batch),
                     epochs=epochs, callbacks=get_callbacks(nombre, guardar=base),
                     verbose=0)
    if base:
        plot_history(hist, nombre)
    res = {
        "Hiperparametros": {"regularizacion": tipo or "ninguna", "l1": l1, "l2": l2,
                            "learning_rate": lr, "batch_size": batch,
                            "epochs_max": epochs, "optimizador": "adam",
                            "loss": "binary_crossentropy"},
        "Epocas ejecutadas": len(hist.history["loss"]),
        "Validacion": evaluate(model, X_val, y_val),
    }
    if HAY_TEST:
        res["Test"] = evaluate(model, X_test, y_test)
    v = res["Validacion"]
    print(f"{nombre} -> Acc: {v['Accuracy']:.4f} | F1: {v['F1-Score']:.4f} | "
          f"AUC: {v['AUC-ROC']:.4f} | Loss: {v['Loss']:.4f} | Epocas: {res['Epocas ejecutadas']}")
    return res


# 5. Configuracion: (nombre_modelo, etiqueta, tipo_reg, l1, l2, lr, batch)
#    La primera variante de cada modelo es la configuracion BASE.
CONFIGS = {
    "Modelo 1 (MLP Base)": ("m1_mlp_base", [
        (None, 0, 0, 0.001, 32),    # base
        (None, 0, 0, 0.01, 32),     # lr mayor
        (None, 0, 0, 0.001, 64),    # batch mayor
    ]),
    "Modelo 2 (L2 Ridge)": ("m2_l2_ridge", [
        ("l2", 0, 0.01, 0.001, 32),  # base
        ("l2", 0, 0.001, 0.001, 32), # lambda menor
        ("l2", 0, 0.1, 0.001, 32),   # lambda mayor
    ]),
    "Modelo 3 (L1 Lasso)": ("m3_l1_lasso", [
        ("l1", 0.005, 0, 0.001, 32),   # base
        ("l1", 0.0005, 0, 0.001, 32),  # lambda menor
        ("l1", 0.05, 0, 0.001, 32),    # lambda mayor
    ]),
    "Modelo 4 (ElasticNet)": ("m4_elasticnet", [
        ("l1_l2", 0.002, 0.005, 0.001, 32),   # base
        ("l1_l2", 0.0002, 0.0005, 0.001, 32), # lambdas menores
        ("l1_l2", 0.01, 0.02, 0.001, 32),     # lambdas mayores
    ]),
}

results = {}       # solo configuraciones base (tabla comparativa)
experimentos = {}  # todas las corridas (analisis de hiperparametros)

for i, (titulo, (nombre, variantes)) in enumerate(CONFIGS.items(), 1):
    print(f"\n[{i}/4] Entrenando {titulo}...")
    for j, (tipo, l1, l2, lr, batch) in enumerate(variantes):
        es_base = (j == 0)
        nombre_run = nombre if es_base else f"{nombre}_exp{j}"
        res = correr(nombre_run, tipo, l1, l2, lr, batch, base=es_base)
        experimentos[nombre_run] = res
        if es_base:
            results[titulo] = res

# 6. Guardar resultados
with open(DATA_DIR / "results_grupo1.json", "w") as f:
    json.dump(results, f, indent=4)
with open(DATA_DIR / "experimentos_grupo1.json", "w") as f:
    json.dump(experimentos, f, indent=4)

print("\nPrimeros 4 modelos entrenados y evaluados con exito!")