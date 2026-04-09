import pyautogui
import cv2
import easyocr
import numpy as np
import pandas as pd
import re
import time

def limpiar_precio(texto):
    # 1. Eliminar todo lo que no sea número, punto, coma o $
    limpio = re.sub(r'[^\d\.\,\$]', '', texto)
    # 2. Si el texto resultante tiene al menos 3 números, es un candidato a precio
    if len(re.findall(r'\d', limpio)) >= 3:
        return limpio
    return None

def extraer_en_vivo_final():
    reader = easyocr.Reader(['es'], gpu=False)
    print("Prepara la pantalla... Tienes 5 segundos.")
    for i in range(5, 0, -1):
        print(f"{i}..."); time.sleep(1)

    # Captura
    screenshot = pyautogui.screenshot()
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    print("Analizando con alta precisión...")
    # Usamos un contraste leve para ayudar al OCR con los números pequeños
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resultados = reader.readtext(img_gray, detail=1)

    productos_lista = []
    textos_generales = []
    precios_encontrados = []

    # PASO 1: Identificar Precios con filtro flexible
    for bbox, texto, conf in resultados:
        precio_candidato = limpiar_precio(texto)
        # Si el texto contiene "$" o tiene formato de miles (X.XXX)
        if precio_candidato and (('$' in precio_candidato) or ('.' in precio_candidato and len(precio_candidato) >= 5)):
            precios_encontrados.append({'bbox': bbox, 'valor': precio_candidato})
        else:
            textos_generales.append({'bbox': bbox, 'texto': texto})

    # PASO 2: Agrupación Espacial
    for p in precios_encontrados:
        px_min, py_min = p['bbox'][0]
        candidatos = []
        
        for t in textos_generales:
            tx_min = t['bbox'][0][0]
            ty_max = t['bbox'][2][1]
            distancia_y = py_min - ty_max
            distancia_x = abs(px_min - tx_min)
            
            # Buscamos en un radio de 200px hacia arriba y 150px hacia los lados
            if 0 < distancia_y < 200 and distancia_x < 150:
                candidatos.append((ty_max, t['texto']))

        candidatos.sort(key=lambda x: x[0])
        
        if candidatos:
            marca = candidatos[0][1]
            # Unimos el resto como nombre del producto
            descripcion = " ".join([c[1] for c in candidatos[1:]]) if len(candidatos) > 1 else marca
            if descripcion == marca: marca = "RETAIL" # Caso donde solo hay una línea
            
            productos_lista.append({
                "Marca": marca.upper(),
                "Producto": descripcion,
                "Precio": p['valor']
            })

    return pd.DataFrame(productos_lista).drop_duplicates()

if __name__ == "__main__":
    df = extraer_en_vivo_final()
    print("\n--- LISTADO DE PRODUCTOS Y PRECIOS ---")
    if not df.empty:
        # Formatear la salida para que se vea como una tabla real
        print(df.to_string(index=False, justify='left'))
        df.to_csv("extraccion_final.csv", index=False)
    else:
        print("No se detectaron precios claros. Prueba subiendo el Zoom del navegador al 125%.")
