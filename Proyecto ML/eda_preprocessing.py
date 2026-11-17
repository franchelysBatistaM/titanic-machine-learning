import os
import json
from pathlib import Path

import pandas as pd  # type: ignore[reportMissingModuleSource]
import numpy as np  # type: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]
import seaborn as sns  # type: ignore[reportMissingModuleSource]
import tensorflow as tf  # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split  # type: ignore[reportMissingModuleSource]

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_processed"
GRAPH_DIR = BASE_DIR / "graficas"
DATASET_DIR = BASE_DIR / "dataset"

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ==========================================
# 1. CREACIÓN DE CARPETAS Y CARGA DE DATOS
# ==========================================
print("Cargando conjuntos de datos...")
DATA_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATASET_DIR / "train.csv")
test_df = pd.read_csv(DATASET_DIR / "test.csv")

print(f" - Entrenamiento: {train_df.shape[0]} filas, {train_df.shape[1]} columnas")
print(f" - Prueba (Kaggle): {test_df.shape[0]} filas, {test_df.shape[1]} columnas\n")

# Guardar PassengerId para la entrega final en Kaggle
np.save(DATA_DIR / "passenger_ids_test.npy", test_df['PassengerId'].values)

# ==========================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# (se hace sobre todo el train.csv: solo exploramos, no entrenamos)
# ==========================================
print("Estadisticas descriptivas y nulos:")
print(train_df.describe().round(2))
print("\nValores nulos por columna:")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])
train_df.describe().round(2).to_csv(DATA_DIR / "eda_estadisticas.csv")

# Outliers por regla del IQR (para explicarlos en el informe)
for col in ['Age', 'Fare']:
    q1, q3 = train_df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((train_df[col] < q1 - 1.5 * iqr) | (train_df[col] > q3 + 1.5 * iqr)).sum()
    print(f"Outliers en {col} (IQR): {n_out}")

print("\nGenerando graficas del EDA...")

# Gráfica 1: Distribución de la variable objetivo
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=train_df, x='Survived', hue='Survived',
                   palette=['#E53E3E', '#3182CE'], legend=False)
plt.title('Distribución de Supervivencia (0 = No, 1 = Sí)')
plt.xlabel('Superviviente')
plt.ylabel('Cantidad de Pasajeros')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10,
                xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.savefig(GRAPH_DIR / "eda_1_distribucion_supervivencia.png", dpi=300)
plt.close()

# Gráfica 2: Supervivencia por género
plt.figure(figsize=(6, 4))
sns.barplot(data=train_df, x='Sex', y='Survived', hue='Sex', palette='Set2', legend=False)
plt.title('Tasa de Supervivencia según el Género')
plt.xlabel('Género')
plt.ylabel('Proporción de Supervivientes')
plt.tight_layout()
plt.savefig(GRAPH_DIR / "eda_2_supervivencia_por_sexo.png", dpi=300)
plt.close()

# Gráfica 3: Supervivencia por clase y género
plt.figure(figsize=(6, 4))
sns.barplot(data=train_df, x='Pclass', y='Survived', hue='Sex', palette='tab10')
plt.title('Supervivencia por Clase (Pclass) y Género')
plt.xlabel('Clase del Pasajero (1 = Primera, 3 = Tercera)')
plt.ylabel('Proporción de Supervivientes')
plt.tight_layout()
plt.savefig(GRAPH_DIR / "eda_3_supervivencia_por_clase_sexo.png", dpi=300)
plt.close()

# Gráfica 4: Distribución de edades
plt.figure(figsize=(7, 4))
sns.histplot(data=train_df, x='Age', hue='Survived', kde=True, element="step",
             palette=['#E53E3E', '#3182CE'])
plt.title('Distribución de Edades según Supervivencia')
plt.xlabel('Edad (Años)')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig(GRAPH_DIR / "eda_4_distribucion_edades.png", dpi=300)
plt.close()

# Gráfica 5: Matriz de correlación (sin PassengerId, que es solo un identificador)
plt.figure(figsize=(8, 6))
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.drop('PassengerId')
sns.heatmap(train_df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm',
            vmin=-1, vmax=1, linewidths=0.5)
plt.title('Matriz de Correlación de Variables Numéricas')
plt.tight_layout()
plt.savefig(GRAPH_DIR / "eda_5_matriz_correlacion.png", dpi=300)
plt.close()

# Gráfica 6: Boxplots para detectar outliers (Age y Fare)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col in zip(axes, ['Age', 'Fare']):
    sns.boxplot(data=train_df, x='Survived', y=col, hue='Survived',
                palette=['#E53E3E', '#3182CE'], legend=False, ax=ax)
    ax.set_title(f'Boxplot de {col} por Supervivencia')
plt.tight_layout()
plt.savefig(GRAPH_DIR / "eda_6_boxplots_outliers.png", dpi=300)
plt.close()

# Gráfica 7: Distribución de Fare (sesgo) antes y después de log1p
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(train_df['Fare'], kde=True, ax=axes[0])
axes[0].set_title('Fare original (muy sesgada)')
sns.histplot(np.log1p(train_df['Fare']), kde=True, ax=axes[1])
axes[1].set_title('Fare con log1p')
plt.tight_layout()
plt.savefig(GRAPH_DIR / "eda_7_fare_log.png", dpi=300)
plt.close()

print("Graficas del EDA guardadas en '/graficas'.\n")

# ==========================================
# 3. INGENIERÍA DE CARACTERÍSTICAS
# (operaciones fila por fila: no usan estadísticos del dataset, no hay fuga)
# ==========================================
TITULOS_VALIDOS = ["Mr", "Mrs", "Miss", "Master"]

def feature_engineering(df):
    out = df.copy()
    title = out['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
    title = title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    out['Title'] = title.where(title.isin(TITULOS_VALIDOS), 'Rare')
    out['FamilySize'] = out['SibSp'] + out['Parch'] + 1
    out['IsAlone'] = (out['FamilySize'] == 1).astype(int)
    out['HasCabin'] = out['Cabin'].notna().astype(int)
    # Se descartan PassengerId, Name, Ticket y Cabin (ya extrajimos lo útil)
    return out[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'Title', 'FamilySize', 'IsAlone', 'HasCabin']]

# ==========================================
# 4. DIVISIÓN TRAIN / VALIDACIÓN / TEST PROPIO (70 / 15 / 15)
# Se divide ANTES de imputar y escalar para evitar fuga de datos.
# ==========================================
print("Dividiendo dataset en Train (70%), Validacion (15%) y Test propio (15%)...")
y_all = train_df['Survived'].values

tmp_raw, hold_raw, y_tmp, y_hold = train_test_split(
    train_df, y_all, test_size=0.15, random_state=SEED, stratify=y_all
)
tr_raw, val_raw, y_train, y_val = train_test_split(
    tmp_raw, y_tmp, test_size=0.15 / 0.85, random_state=SEED, stratify=y_tmp
)

df_train = feature_engineering(tr_raw)
df_val = feature_engineering(val_raw)
df_hold = feature_engineering(hold_raw)
df_kaggle = feature_engineering(test_df)

print(f" - Train: {len(df_train)} | Val: {len(df_val)} | Test propio: {len(df_hold)} "
      f"| Test Kaggle (sin etiquetas): {len(df_kaggle)}")

# ==========================================
# 5. LIMPIEZA: imputación con estadísticos calculados SOLO con train
# ==========================================
age_median = float(df_train['Age'].median())
fare_median = float(df_train['Fare'].median())
embarked_mode = str(df_train['Embarked'].mode()[0])

def imputar(df):
    out = df.copy()
    out['Age'] = out['Age'].fillna(age_median)
    out['Fare'] = out['Fare'].fillna(fare_median)
    out['Embarked'] = out['Embarked'].fillna(embarked_mode)
    out['Fare'] = np.log1p(out['Fare'])  # reduce el sesgo y el efecto de outliers
    return out

df_train, df_val, df_hold, df_kaggle = map(imputar, [df_train, df_val, df_hold, df_kaggle])
print("Valores nulos imputados con estadisticos de train.")

# ==========================================
# 6. CODIFICACIÓN (One-Hot) Y ESCALADO (tf.keras.layers.Normalization)
# ==========================================
CAT_COLS = ['Sex', 'Embarked', 'Title']
NUM_COLS = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']

train_enc = pd.get_dummies(df_train, columns=CAT_COLS, dtype=float)
feature_names = list(train_enc.columns)

def codificar(df):
    enc = pd.get_dummies(df, columns=CAT_COLS, dtype=float)
    return enc.reindex(columns=feature_names, fill_value=0.0)

val_enc, hold_enc, kaggle_enc = map(codificar, [df_val, df_hold, df_kaggle])

# La capa Normalization aprende media y varianza SOLO con train (.adapt)
norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(train_enc[NUM_COLS].values.astype("float32"))

def escalar(df):
    out = df.copy()
    out[NUM_COLS] = norm(df[NUM_COLS].values.astype("float32")).numpy()
    return out.values.astype("float32")

X_train = escalar(train_enc)
X_val = escalar(val_enc)
X_holdout = escalar(hold_enc)
X_test = escalar(kaggle_enc)

# ==========================================
# 7. EXPORTACIÓN DE ARCHIVOS
# ==========================================
print("\nGuardando archivos procesados en '/data_processed'...")
np.save(DATA_DIR / "X_train.npy", X_train)
np.save(DATA_DIR / "X_val.npy", X_val)
np.save(DATA_DIR / "X_holdout.npy", X_holdout)
np.save(DATA_DIR / "X_test.npy", X_test)          # test de Kaggle (sin etiquetas)
np.save(DATA_DIR / "y_train.npy", y_train)
np.save(DATA_DIR / "y_val.npy", y_val)
np.save(DATA_DIR / "y_holdout.npy", y_hold)

with open(DATA_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)

# Parámetros para repetir EXACTAMENTE el mismo preprocesamiento con un CSV nuevo
params = {
    "age_median": age_median,
    "fare_median": fare_median,
    "embarked_mode": embarked_mode,
    "titulos_validos": TITULOS_VALIDOS,
    "cat_cols": CAT_COLS,
    "num_cols": NUM_COLS,
    "fare_log1p": True,
    "columnas_finales": feature_names,
    "norm_mean": norm.mean.numpy().ravel().tolist(),
    "norm_variance": norm.variance.numpy().ravel().tolist(),
}
with open(DATA_DIR / "preprocessing_params.json", "w") as f:
    json.dump(params, f, indent=2)

print(f"Features finales: {len(feature_names)} -> {feature_names}")
print("PREPROCESAMIENTO COMPLETADO CON EXITO!")