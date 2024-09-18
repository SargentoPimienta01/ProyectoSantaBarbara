import cv2
import numpy as np

# Función para extraer los colores de referencia de una imagen capturada
def extraer_colores_referencia(imagen, puntos_referencia):
    colores_detectados = []
    for punto in puntos_referencia:
        # Extraemos el color del punto de referencia
        color = imagen[punto[1], punto[0]]
        colores_detectados.append(color)
    return np.array(colores_detectados)

# Función para calcular la matriz de transformación de color
def calcular_matriz_transformacion(colores_reales, colores_detectados):
    # Convertimos los colores a float para la transformación
    colores_reales = np.array(colores_reales, dtype=np.float32)
    colores_detectados = np.array(colores_detectados, dtype=np.float32)
    
    # Usamos la función cv2.findHomography para encontrar la transformación entre los colores reales y los detectados
    matriz_transformacion, _ = cv2.findHomography(colores_detectados, colores_reales)
    
    return matriz_transformacion

# Función para aplicar la calibración de color a la imagen
def aplicar_calibracion(imagen, matriz_transformacion):
    # Aplicamos la transformación de color usando la matriz calculada
    altura, anchura = imagen.shape[:2]
    imagen_calibrada = cv2.warpPerspective(imagen, matriz_transformacion, (anchura, altura))
    
    return imagen_calibrada

# Función para calibrar una imagen
def calibrar_imagen(imagen, puntos_referencia, colores_reales):
    # Extraemos los colores de los puntos de referencia de la imagen capturada
    colores_detectados = extraer_colores_referencia(imagen, puntos_referencia)
    
    # Calculamos la matriz de transformación de color
    matriz_transformacion = calcular_matriz_transformacion(colores_reales, colores_detectados)
    
    # Aplicamos la calibración a la imagen
    imagen_calibrada = aplicar_calibracion(imagen, matriz_transformacion)
    
    return imagen_calibrada

if __name__ == "__main__":
    # Cargar la imagen capturada por la cámara
    imagen = cv2.imread('captura_cam.jpg')

    # Definir los puntos de referencia de la carta de color (en píxeles)
    puntos_referencia = [
        (100, 200),  # Coordenada del primer color en la carta de colores
        (200, 200),  # Coordenada del segundo color
        (300, 200),  # Coordenada del tercer color
        # Agregar más puntos según el patrón de referencia
    ]

    # Valores RGB reales de la carta de colores (patrón de referencia conocido)
    colores_reales = [
        [115, 82, 68],   # Color de referencia 1 (en RGB)
        [194, 150, 130], # Color de referencia 2
        [98, 122, 157],  # Color de referencia 3
        # Agregar más colores reales correspondientes a los puntos de referencia
    ]

    # Calibrar la imagen
    imagen_calibrada = calibrar_imagen(imagen, puntos_referencia, colores_reales)

    # Mostrar la imagen original y la imagen calibrada
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Imagen Calibrada', imagen_calibrada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
