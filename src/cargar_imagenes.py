import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Directorio donde están almacenadas las imágenes
data_dir = '../data/'

# Función de colorimetría para análisis en espacio HSV
def colorimetria(directorio, tamaño=(128, 128)):
    imagenes = []
    for archivo in os.listdir(directorio):
        if archivo.endswith('.jpg') or archivo.endswith('.png'):
            ruta = os.path.join(directorio, archivo)
            imagen = cv2.imread(ruta)
            if imagen is not None:
                # Redimensionar la imagen
                imagen = cv2.resize(imagen, tamaño)
                
                # Convertir la imagen de BGR a HSV para análisis de color
                imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
                
                # Extraer el canal de la tonalidad (Hue) para analizar la colorimetría
                hue_channel = imagen_hsv[:, :, 0]  # Canal H
                
                # Agregar la imagen preprocesada y el análisis de tonalidad
                imagenes.append({
                    "original": imagen,
                    "hsv": imagen_hsv,
                    "hue": hue_channel
                })
    return imagenes

# Mostrar histograma de la tonalidad (Hue)
def mostrar_histograma_color(imagenes):
    for i, img_data in enumerate(imagenes):
        hue_channel = img_data["hue"]
        plt.hist(hue_channel.ravel(), bins=180, range=(0, 180))
        plt.title(f'Histograma de Hue - Imagen {i+1}')
        plt.xlabel('Hue')
        plt.ylabel('Frecuencia')
        plt.show()

# Función para cargar y preprocesar imágenes en escala de grises
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

# Mostrar las imágenes preprocesadas en escala de grises
def mostrar_imagenes(imagenes):
    for i, imagen in enumerate(imagenes):
        plt.imshow(imagen, cmap='gray')
        plt.title(f'Imagen {i+1}')
        plt.show()

if __name__ == '__main__':
    # Colorimetría
    imagenes = colorimetria(data_dir)
    if imagenes:
        mostrar_histograma_color(imagenes)
    else:
        print("No se encontraron imágenes.")
    
    # Preprocesamiento en escala de grises
    imagenes = cargar_y_preprocesar_imagenes(data_dir)
    if imagenes.any():
        mostrar_imagenes(imagenes)
    else:
        print("No se encontraron imágenes.")
