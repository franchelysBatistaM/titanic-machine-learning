import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configurar estilo visual de las gráficas
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

# ==========================================
# 1. CREACIÓN DE CARPETAS Y CARGA DE DATOS
# ==========================================
print("📥 Cargando conjuntos de datos...")
os.makedirs("data_processed", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

print("✅ Datos cargados correctamente.")
print(f" - Entrenamiento: {train_df.shape[0]} filas, {train_df.shape[1]} columnas")
print(f" - Prueba (Test): {test_df.shape[0]} filas, {test_df.shape[1]} columnas\n")

# Guardar PassengerId para la entrega final en Kaggle
passenger_ids_test = test_df['PassengerId'].values
np.save("data_processed/passenger_ids_test.npy", passenger_ids_test)

# ==========================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==========================================
print("📊 Generando gráficas del EDA para el Informe...")

# Gráfica 1: Distribución de la variable objetivo (Survived)
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=train_df, x='Survived', palette=['#E53E3E', '#3182CE'])
plt.title('Distribución de Supervivencia (0 = No, 1 = Sí)')
plt.xlabel('Superviviente')
plt.ylabel('Cantidad de Pasajeros')
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom', fontsize=10,
        xytext=(0, 3), textcoords='offset points'
    )
plt.tight_layout()
plt.savefig("graficas/eda_1_distribucion_supervivencia.png", dpi=300)
plt.close()

# Gráfica 2: Tasa de Supervivencia por Género
plt.figure(figsize=(6, 4))
sns.barplot(data=train_df, x='Sex', y='Survived', palette='Set2')
plt.title('Tasa de Supervivencia según el Género')
plt.xlabel('Género')
plt.ylabel('Proporción de Supervivientes')
plt.tight_layout()
plt.savefig("graficas/eda_2_supervivencia_por_sexo.png", dpi=300)
plt.close()

# Gráfica 3: Tasa de Supervivencia por Clase de Ticket (Pclass)
plt.figure(figsize=(6, 4))
sns.barplot(data=train_df, x='Pclass', y='Survived', hue='Sex', palette='tab10')
plt.title('Supervivencia por Clase (Pclass) y Género')
plt.xlabel('Clase del Pasajero (1 = Primera, 3 = Tercera)')
plt.ylabel('Proporción de Supervivientes')
plt.tight_layout()
plt.savefig("graficas/eda_3_supervivencia_por_clase_sexo.png", dpi=300)
plt.close()

# Gráfica 4: Distribución de Edades
plt.figure(figsize=(7, 4))
sns.histplot(
    data=train_df, x='Age', hue='Survived',
    kde=True, element="step", palette=['#E53E3E', '#3182CE']
)
plt.title('Distribución de Edades según Supervivencia')
plt.xlabel('Edad (Años)')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig("graficas/eda_4_distribucion_edades.png", dpi=300)
plt.close()

# Gráfica 5: Matriz de Correlación
plt.figure(figsize=(8, 6))
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
corr = train_df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Matriz de Correlación de Variables Numéricas')
plt.tight_layout()
plt.savefig("graficas/eda_5_matriz_correlacion.png", dpi=300)
plt.close()

print("✅ Gráficas del EDA guardadas exitosamente en la carpeta '/graficas'.\n")

# ==========================================
# 3. LIMPIEZA Y MANEJO DE VALORES NULOS
# ==========================================
print("🧹 Limpiando datos y rellenando valores nulos...")
y = train_df['Survived'].values
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']

df_train_clean = train_df.drop(columns=drop_cols)
df_test_clean = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])

# Imputación de nulos
age_median = df_train_clean['Age'].median()
fare_median = df_train_clean['Fare'].median()
embarked_mode = df_train_clean['Embarked'].mode()[0]

for df in [df_train_clean, df_test_clean]:
    df['Age'] = df['Age'].fillna(age_median)
    df['Fare'] = df['Fare'].fillna(fare_median)
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)

print("✅ Valores nulos imputados correctamente.")

# ==========================================
# 4. CODIFICACIÓN Y ESCALADO
# ==========================================
print("⚙️ Codificando variables categóricas y escalando numéricas...")
df_train_encoded = pd.get_dummies(df_train_clean, columns=['Sex', 'Embarked'], drop_first=True, dtype=float)
df_test_encoded = pd.get_dummies(df_test_clean, columns=['Sex', 'Embarked'], drop_first=True, dtype=float)
df_test_encoded = df_test_encoded.reindex(columns=df_train_encoded.columns, fill_value=0.0)

feature_names = list(df_train_encoded.columns)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train_encoded)
X_test_scaled = scaler.transform(df_test_encoded)

# ==========================================
# 5. DIVISIÓN TRAIN (80%) Y VALIDACIÓN (20%)
# ==========================================
print("✂️ Dividiendo dataset en Train (80%) y Validación (20%)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================
# 6. EXPORTACIÓN DE ARCHIVOS
# ==========================================
print("\n💾 Guardando archivos procesados en '/data_processed'...")
np.save("data_processed/X_train.npy", X_train)
np.save("data_processed/X_val.npy", X_val)
np.save("data_processed/y_train.npy", y_train)
np.save("data_processed/y_val.npy", y_val)
np.save("data_processed/X_test.npy", X_test_scaled)

with open("data_processed/feature_names.json", "w") as f:
    json.dump(feature_names, f)

print("🎉 ¡PREPROCESAMIENTO COMPLETADO CON ÉXITO!")
