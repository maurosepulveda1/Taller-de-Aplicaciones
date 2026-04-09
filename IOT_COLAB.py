import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

# 1. Carga de datos ultra-robusta
try:
    df = pd.read_csv('/content/Acelerometro V2.csv', sep=';', decimal='.')

    df.columns = df.columns.str.strip()

    col_x = 'Linear Acceleration x (m/s^2)'
    col_y = 'Linear Acceleration y (m/s^2)'
    col_z = 'Linear Acceleration z (m/s^2)'
    col_abs = 'Absolute acceleration (m/s^2)'
    col_tipo = 'tipo'

    # Conversión forzada a numérico (por si acaso leyó algo como texto)
    for col in [col_x, col_y, col_z, col_abs]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Eliminar filas con errores si existieran
    df = df.dropna()

except Exception as e:
    print(f"Error crítico al leer el archivo: {e}")

# 2. Ingeniería de Características por Ventanas
def extraer_caracteristicas(data, window_size=50): # Ventana más pequeña para capturar más detalle
    features = []
    labels = []

    for i in range(0, len(data) - window_size, window_size):
        ventana = data.iloc[i : i + window_size]

        # Estadísticas que definen el "movimiento"
        f = [
            ventana[col_abs].mean(),
            ventana[col_abs].std(),
            ventana[col_x].std(),
            ventana[col_y].std(),
            ventana[col_z].std(),
            ventana[col_abs].max() - ventana[col_abs].min()
        ]

        features.append(f)
        labels.append(ventana[col_tipo].mode()[0])

    return np.array(features), np.array(labels)

# 3. Procesamiento y Modelado
try:
    X_raw, y_real = extraer_caracteristicas(df)

    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Reducción de dimensionalidad (t-SNE)
    print("Calculando t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(X_raw)-1), random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Agrupamiento
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # 4. Visualización
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_real, palette='Set1', s=60)
    plt.title("Realidad (Etiquetas del archivo)")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters, palette='viridis', s=60)
    plt.title("Agrupamiento Sugerido (K-Means)")

    plt.show()

    # Métrica
    print(f"\nÍndice Rand Ajustado: {adjusted_rand_score(y_real, clusters):.4f}")
    print("\nCruce de datos (Matriz):")
    print(pd.crosstab(y_real, clusters))

except Exception as e:
    print(f"Error en el proceso: {e}")