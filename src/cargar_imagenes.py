import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Directorio donde están almacenadas las imágenes
data_dir = '../data/'

# Función para cargar y preprocesar imágenes
def cargar_y_preprocesar_imagenes(directorio, tamaño=(128, 128)):
    imagenes = []
    for archivo in os.listdir(directorio):
        if archivo.endswith('.jpg') or archivo.endswith('.png'):
            ruta = os.path.join(directorio, archivo)
            imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises
            if imagen is not None:
                # Redimensionar la imagen
                imagen = cv2.resize(imagen, tamaño)
                # Normalizar los valores de los píxeles
                imagen = imagen / 255.0
                imagenes.append(imagen)
    return np.array(imagenes)

# Mostrar las imágenes
def mostrar_imagenes(imagenes):
    for i, imagen in enumerate(imagenes):
        plt.imshow(imagen, cmap='gray')
        plt.title(f'Imagen {i+1}')
        plt.show()

if __name__ == '__main__':
    imagenes = cargar_y_preprocesar_imagenes(data_dir)
    if imagenes.any():
        mostrar_imagenes(imagenes)
    else:
        print("No se encontraron imágenes.")
