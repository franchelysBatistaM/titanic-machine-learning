import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # pyright: ignore[reportMissingModuleSource]
from tensorflow import keras  # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras import layers, callbacks  # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_processed"
MODEL_DIR = BASE_DIR / "saved_models"
GRAPH_DIR = BASE_DIR / "graficas"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos procesados por eda_preprocessing.py (mismos que el grupo 1)
X_train = np.load(DATA_DIR / "X_train.npy").astype("float32")
X_val = np.load(DATA_DIR / "X_val.npy").astype("float32")
y_train = np.load(DATA_DIR / "y_train.npy").astype("float32")
y_val = np.load(DATA_DIR / "y_val.npy").astype("float32")
X_test = np.load(DATA_DIR / "X_holdout.npy").astype("float32")   # test propio con etiquetas
y_test = np.load(DATA_DIR / "y_holdout.npy").astype("float32")

n_features = X_train.shape[1]
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape} | Features: {n_features}")


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


# 3. Evaluacion completa (igual que en el grupo 1 para poder comparar)
def evaluate(model, X, y):
    probs = model.predict(make_ds(X, y, 128), verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
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


def compilar(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model


# 4. Arquitecturas
def build_dropout(rate=0.3, lr=0.001):
    """Modelo 5: MLP con Dropout despues de cada capa oculta."""
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(rate),
        layers.Dense(32, activation='relu'),
        layers.Dropout(rate),
        layers.Dense(1, activation='sigmoid')
    ])
    return compilar(model, lr)


def build_batchnorm(lr=0.001):
    """Modelo 6: MLP con BatchNormalization (Dense -> BN -> ReLU)."""
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return compilar(model, lr)


def build_conv1d(filters=32, kernel=3, lr=0.001):
    """Modelo 8: Conv1D tratando las features como una secuencia de longitud n_features."""
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Reshape((n_features, 1)),                       # (batch, features, 1 canal)
        layers.Conv1D(filters, kernel, padding='same', activation='relu'),
        layers.Conv1D(filters * 2, kernel, padding='same', activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return compilar(model, lr)


def build_autoencoder(latent=8, lr=0.001):
    """Modelo 7 (parte 1): Autoencoder  n_features -> 16 -> latent -> 16 -> n_features."""
    inp = keras.Input(shape=(n_features,))
    x = layers.Dense(16, activation='relu')(inp)
    z = layers.Dense(latent, activation='relu', name='latente')(x)
    x = layers.Dense(16, activation='relu')(z)
    out = layers.Dense(n_features, activation='linear')(x)
    autoenc = keras.Model(inp, out, name='autoencoder')
    encoder = keras.Model(inp, z, name='encoder')
    autoenc.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return autoenc, encoder


# 5. Resumen de resultados
def armar_resultado(model, hiper, epocas, extra=None):
    res = {
        "Hiperparametros": hiper,
        "Epocas ejecutadas": epocas,
        "Validacion": evaluate(model, X_val, y_val),
        "Test": evaluate(model, X_test, y_test),
    }
    if extra:
        res.update(extra)
    return res


def imprimir(nombre, res):
    v = res["Validacion"]
    print(f"{nombre} -> Acc: {v['Accuracy']:.4f} | F1: {v['F1-Score']:.4f} | "
          f"AUC: {v['AUC-ROC']:.4f} | Loss: {v['Loss']:.4f} | Epocas: {res['Epocas ejecutadas']}")


def correr(nombre, builder, hiper, lr, batch, epochs=80, base=False):
    """Entrena un modelo supervisado normal (modelos 5, 6 y 8)."""
    keras.utils.set_random_seed(SEED)   # misma semilla para comparar justo
    model = builder()
    hist = model.fit(make_ds(X_train, y_train, batch, shuffle=True),
                     validation_data=make_ds(X_val, y_val, batch),
                     epochs=epochs, callbacks=get_callbacks(nombre, guardar=base),
                     verbose=0)
    if base:
        plot_history(hist, nombre)
    hp = dict(hiper)
    hp.update({"learning_rate": lr, "batch_size": batch, "epochs_max": epochs,
               "optimizador": "adam", "loss": "binary_crossentropy"})
    res = armar_resultado(model, hp, len(hist.history["loss"]))
    imprimir(nombre, res)
    return res


def correr_autoencoder(nombre, latent, lr, batch, epochs=80, base=False):
    """Modelo 7: 1) entrena el autoencoder sin etiquetas, 2) congela el encoder
    y entrena un clasificador sobre las features comprimidas."""
    keras.utils.set_random_seed(SEED)
    autoenc, encoder = build_autoencoder(latent, lr)

    # Fase 1: reconstruccion (X -> X). No usa y.
    hist_ae = autoenc.fit(X_train, X_train, validation_data=(X_val, X_val),
                          epochs=150, batch_size=batch, verbose=0,
                          callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                                             restore_best_weights=True),
                                     callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                                 patience=5, min_lr=1e-5)])
    if base:
        plt.figure(figsize=(6, 4))
        plt.plot(hist_ae.history['loss'], label='Entrenamiento')
        plt.plot(hist_ae.history['val_loss'], label='Validación')
        plt.title(f'Reconstrucción del Autoencoder (MSE) - {nombre}')
        plt.xlabel('Época')
        plt.legend()
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / f"curvas_{nombre}_reconstruccion.png", dpi=300)
        plt.close()

    # Fase 2: clasificador sobre el espacio latente (encoder congelado)
    encoder.trainable = False
    inp = keras.Input(shape=(n_features,))
    z = encoder(inp)
    x = layers.Dense(16, activation='relu')(z)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = compilar(keras.Model(inp, out, name='autoencoder_clasificador'), lr)

    hist = model.fit(make_ds(X_train, y_train, batch, shuffle=True),
                     validation_data=make_ds(X_val, y_val, batch),
                     epochs=epochs, callbacks=get_callbacks(nombre, guardar=base),
                     verbose=0)
    if base:
        plot_history(hist, nombre)

    hp = {"dim_latente": latent, "encoder": f"{n_features}-16-{latent}",
          "learning_rate": lr, "batch_size": batch, "epochs_max": epochs,
          "optimizador": "adam", "loss": "binary_crossentropy (clasificador) / mse (autoencoder)"}
    res = armar_resultado(model, hp, len(hist.history["loss"]),
                          extra={"Epocas autoencoder": len(hist_ae.history["loss"]),
                                 "MSE reconstruccion (val)": float(min(hist_ae.history["val_loss"]))})
    imprimir(nombre, res)
    return res


# 6. Configuracion: la primera variante de cada modelo es la configuracion BASE
results = {}       # solo configuraciones base (tabla comparativa)
experimentos = {}  # todas las corridas (analisis de hiperparametros)


def registrar(titulo, nombre_run, res, es_base):
    experimentos[nombre_run] = res
    if es_base:
        results[titulo] = res


# --- Modelo 5: Dropout ---
print("\n[1/4] Entrenando Modelo 5 (Dropout)...")
for j, (rate, lr, batch) in enumerate([(0.3, 0.001, 32),    # base
                                       (0.1, 0.001, 32),    # dropout menor
                                       (0.5, 0.001, 32)]):  # dropout mayor
    nombre = "m5_dropout" if j == 0 else f"m5_dropout_exp{j}"
    res = correr(nombre, lambda r=rate, l=lr: build_dropout(r, l),
                 {"dropout_rate": rate}, lr, batch, base=(j == 0))
    registrar("Modelo 5 (Dropout)", nombre, res, j == 0)

# --- Modelo 6: Batch Normalization ---
print("\n[2/4] Entrenando Modelo 6 (BatchNorm)...")
for j, (lr, batch) in enumerate([(0.001, 32),    # base
                                 (0.01, 32),     # lr mayor
                                 (0.001, 64)]):  # batch mayor
    nombre = "m6_batchnorm" if j == 0 else f"m6_batchnorm_exp{j}"
    res = correr(nombre, lambda l=lr: build_batchnorm(l),
                 {"batch_normalization": True}, lr, batch, base=(j == 0))
    registrar("Modelo 6 (BatchNorm)", nombre, res, j == 0)

# --- Modelo 7: Autoencoder + clasificador ---
print("\n[3/4] Entrenando Modelo 7 (Autoencoder)...")
for j, (latent, lr, batch) in enumerate([(8, 0.001, 32),    # base
                                         (4, 0.001, 32),    # cuello mas estrecho
                                         (12, 0.001, 32)]): # cuello mas ancho
    nombre = "m7_autoencoder" if j == 0 else f"m7_autoencoder_exp{j}"
    res = correr_autoencoder(nombre, latent, lr, batch, base=(j == 0))
    registrar("Modelo 7 (Autoencoder)", nombre, res, j == 0)

# --- Modelo 8: Conv1D ---
print("\n[4/4] Entrenando Modelo 8 (Conv1D)...")
for j, (filters, kernel, lr, batch) in enumerate([(32, 3, 0.001, 32),    # base
                                                  (16, 3, 0.001, 32),    # menos filtros
                                                  (32, 5, 0.001, 32)]):  # kernel mayor
    nombre = "m8_conv1d" if j == 0 else f"m8_conv1d_exp{j}"
    res = correr(nombre, lambda f=filters, k=kernel, l=lr: build_conv1d(f, k, l),
                 {"filtros": filters, "kernel_size": kernel}, lr, batch, base=(j == 0))
    registrar("Modelo 8 (Conv1D)", nombre, res, j == 0)

# 7. Guardar resultados
with open(DATA_DIR / "results_grupo2.json", "w") as f:
    json.dump(results, f, indent=4)
with open(DATA_DIR / "experimentos_grupo2.json", "w") as f:
    json.dump(experimentos, f, indent=4)

print("\nModelos 5 al 8 entrenados y evaluados con exito!")