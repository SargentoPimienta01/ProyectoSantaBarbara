import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Directorio donde están almacenadas las imágenes
data_dir = '../data/'

# Función para detectar huevos en una imagen usando contornos
def detectar_huevos(directorio, tamaño=(128, 128)):
    huevos_detectados = []
    for archivo in os.listdir(directorio):
        if archivo.endswith('.jpg') or archivo.endswith('.png'):
            ruta = os.path.join(directorio, archivo)
            imagen = cv2.imread(ruta)
            if imagen is not None:
                # Convertir la imagen a escala de grises
                imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

                # Aplicar suavizado para eliminar ruido
                imagen_gris = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

                # Detectar bordes usando Canny Edge Detection
                bordes = cv2.Canny(imagen_gris, 50, 150)

                # Encontrar contornos en la imagen
                contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contornos) == 0:
                    print(f"No se detectaron huevos en la imagen {archivo}.")
                    continue  # Pasar a la siguiente imagen si no hay huevos

                for contorno in contornos:
                    # Crear un rectángulo alrededor de cada contorno (huevo)
                    x, y, w, h = cv2.boundingRect(contorno)
                    huevo = imagen[y:y + h, x:x + w]

                    # Redimensionar el huevo a un tamaño fijo
                    huevo = cv2.resize(huevo, tamaño)

                    # Normalizar la imagen
                    huevo = huevo / 255.0

                    # Guardar el huevo detectado
                    huevos_detectados.append(huevo)

    if len(huevos_detectados) == 0:
        print("No se detectaron huevos en ninguna imagen.")
    return np.array(huevos_detectados)

# Función de colorimetría para análisis en espacio HSV
def colorimetria(imagen, tamaño=(128, 128)):
    # Redimensionar la imagen
    imagen = cv2.resize(imagen, tamaño)
    
    # Convertir la imagen de BGR a HSV para análisis de color
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    # Extraer el canal de la tonalidad (Hue) para analizar la colorimetría
    hue_channel = imagen_hsv[:, :, 0]  # Canal H

    return hue_channel

# Mostrar histograma de la tonalidad (Hue)
def mostrar_histograma_color(imagenes):
    for i, img_data in enumerate(imagenes):
        hue_channel = img_data["hue"]
        plt.hist(hue_channel.ravel(), bins=180, range=(0, 180))
        plt.title(f'Histograma de Hue - Imagen {i+1}')
        plt.xlabel('Hue')
        plt.ylabel('Frecuencia')
        plt.show()

def cargar_y_preprocesar_imagenes(directorio, tamaño=(128, 128)):
    imagenes = []
    for archivo in os.listdir(directorio):
        if archivo.endswith('.jpg') or archivo.endswith('.png'):
            ruta = os.path.join(directorio, archivo)
            imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE) 
            if imagen is not None:
                imagen = cv2.resize(imagen, tamaño)
                imagen = imagen / 255.0
                imagenes.append(imagen)
    return np.array(imagenes)

# Mostrar los huevos detectados
def mostrar_huevos(huevos):
    for i, huevo in enumerate(huevos):
        plt.imshow(cv2.cvtColor(huevo, cv2.COLOR_BGR2RGB))
        plt.title(f'Huevo {i+1}')
        plt.show()

# Mostrar las imágenes preprocesadas en escala de grises
def mostrar_imagenes(imagenes):
    for i, imagen in enumerate(imagenes):
        plt.imshow(imagen, cmap='gray')
        plt.title(f'Imagen {i+1}')
        plt.show()

if __name__ == '__main__':
    # Detección de huevos
    huevos_detectados = detectar_huevos(data_dir)
    if len(huevos_detectados) > 0:
        mostrar_huevos(huevos_detectados)
    else:
        print("No se detectaron huevos.")

    # Colorimetría (solo se ejecuta si hay huevos detectados)
    if len(huevos_detectados) > 0:
        for huevo in huevos_detectados:
            hue_channel = colorimetria(huevo)
            plt.hist(hue_channel.ravel(), bins=180, range=(0, 180))
            plt.title('Histograma de Hue')
            plt.xlabel('Hue')
            plt.ylabel('Frecuencia')
            plt.show()

    # Preprocesamiento en escala de grises
    imagenes = cargar_y_preprocesar_imagenes(data_dir)
    if imagenes.any():
        mostrar_imagenes(imagenes)
    else:
        print("No se encontraron imágenes.")
