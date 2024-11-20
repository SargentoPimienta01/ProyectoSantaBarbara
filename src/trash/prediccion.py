import cv2
import numpy as np
from trash.cargar_imagenes import detectar_huevos
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model('ruta_al_modelo/modelo_cnn.h5')

# Función para preprocesar una imagen de prueba
def preprocesar_imagen(ruta_imagen):
    imagen = detectar_huevos(ruta_imagen)
    imagen = np.expand_dims(imagen, axis=0)
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    imagen = cv2.resize(imagen, (128, 128))
    imagen = imagen / 255.0
    imagen = np.expand_dims(imagen, axis=0)  # Añadir la dimensión necesaria
    return imagen

# Cargar y preprocesar una nueva imagen de prueba
imagen_prueba = preprocesar_imagen('../data/Prueba/Prueba02.jpeg')

# Hacer la predicción
prediccion = model.predict(imagen_prueba)
print(f'Probabilidad de que el huevo sea viable: {prediccion[0][0]:.2f}')
