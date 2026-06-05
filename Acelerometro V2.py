# ============================================================
# LABORATORIO 2 — Clasificador de actividades con acelerómetro
# Datos IoT capturados con Phyphox
# Google Colab ready
# ============================================================

# ── 1. INSTALACIÓN Y LIBRERÍAS ─────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── 2. CARGA DE DATOS Y PREPROCESAMIENTO INICIAL ───────────
# Opción A: usar los datos de muestra de arriba
df = pd.read_csv("/content/Acelerometro V2.csv", sep=";")

# Opción B: cargar tu propio archivo desde Google Drive (descomentar)
# from google.colab import files
# uploaded = files.upload()
# import io
# df = pd.read_csv(io.BytesIO(list(uploaded.values())[0]), sep=";")

# Renombrar columnas para mayor comodidad
df.columns = ["time", "ax", "ay", "az", "abs_acc", "tipo"]

# Convertir las columnas de aceleración a tipo numérico, forzando errores a NaN
# y luego eliminando las filas con NaN para asegurar datos limpios y numéricos.
for col in ["ax", "ay", "az", "abs_acc"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=["ax", "ay", "az", "abs_acc"], inplace=True)

print("=== Vista previa de los datos ===")
print(df.head(10))
print(f"\nForma del dataset: {df.shape}")
print(f"\nDistribución de actividades:\n{df['tipo'].value_counts()}")

# ── 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA) ────────────────
fig_eda, axes_eda = plt.subplots(1, 2, figsize=(12, 5))
fig_eda.suptitle("Laboratorio 2 — Análisis Exploratorio de Datos", fontsize=13)

# Distribución de actividades
ax1_eda = axes_eda[0]
counts = df["tipo"].value_counts()
colors = ["#2196F3", "#FF5722", "#4CAF50", "#607D8B"] # Added one more color for 4 activities
ax1_eda.bar(counts.index, counts.values, color=colors[:len(counts)]) # Use colors based on actual counts
ax1_eda.set_title("Distribución de actividades")
ax1_eda.set_xlabel("Actividad")
ax1_eda.set_ylabel("Registros")

# Aceleración absoluta por actividad (boxplot)
ax2_eda = axes_eda[1]
activities = df["tipo"].unique()
data_box = [df[df["tipo"] == act]["abs_acc"].values for act in activities]
bp = ax2_eda.boxplot(data_box, labels=activities, patch_artist=True)
for patch, color in zip(bp["boxes"], colors[:len(activities)]): # Use colors based on actual activities
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2_eda.set_title("Aceleración absoluta por actividad")
ax2_eda.set_ylabel("Aceleración (m/s²)")

plt.tight_layout()
plt.savefig("eda_initial_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGráficos EDA guardados como 'eda_initial_plots.png'")

# ── 4. INGENIERÍA DE CARACTERÍSTICAS (ventanas) ────────────
# El paso 4 del laboratorio pide agregar datos en ventanas de X registros.
# Usamos ventana de 10 registros para calcular media y desviación estándar.

WINDOW_SIZE = 10

features_list = []

for activity in df["tipo"].unique():
    subset = df[df["tipo"] == activity].reset_index(drop=True)
    for start in range(0, len(subset) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = subset.iloc[start:start + WINDOW_SIZE]
        features = {
            # Media de cada eje
            "ax_mean": window["ax"].mean(),
            "ay_mean": window["ay"].mean(),
            "az_mean": window["az"].mean(),
            "abs_mean": window["abs_acc"].mean(),
            # Desviación estándar de cada eje
            "ax_std": window["ax"].std(),
            "ay_std": window["ay"].std(),
            "az_std": window["az"].std(),
            "abs_std": window["abs_acc"].std(),
            # Rango (max - min) como indicador de variabilidad
            "ax_range": window["ax"].max() - window["ax"].min(),
            "ay_range": window["ay"].max() - window["ay"].min(),
            "az_range": window["az"].max() - window["az"].min(),
            "abs_range": window["abs_acc"].max() - window["abs_acc"].min(),
            # Etiqueta
            "tipo": activity
        }
        features_list.append(features)

df_features = pd.DataFrame(features_list)
print(f"\n=== Dataset con ventanas ===")
print(df_features.head())
print(f"Forma: {df_features.shape}")

# ── 5. PREPARACIÓN PARA CLASIFICACIÓN ─────────────────────
X = df_features.drop(columns=["tipo"])
y = df_features["tipo"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

print(f"\nEntrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba:        {X_test.shape[0]} muestras")

# ── 6. CLASIFICADOR — Random Forest y Evaluación ──────────
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== RESULTADOS DEL CLASIFICADOR ===")
print(f"Exactitud (accuracy): {accuracy_score(y_test, y_pred):.2%}")
print("\nReporte detallado:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Visualizaciones del Clasificador

# Matriz de confusión
fig_cm = plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=le.classes_, yticklabels=le.classes_
)
plt.title("Matriz de confusión (Random Forest)")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("matriz_confusion_rf.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGráfico de Matriz de Confusión guardado como 'matriz_confusion_rf.png'")

# Importancia de Variables
feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
feat_imp.tail(10).plot(kind="barh", color="#3F51B5")
plt.title("Top 10 variables más importantes (Random Forest)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.savefig("importancia_variables_rf.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico de Importancia de Variables guardado como 'importancia_variables_rf.png'")

print("\n✅ Laboratorio completado.")