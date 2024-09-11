import tensorflow as tf
from keras import layers, models
import numpy as np
from cargar_imagenes import cargar_y_preprocesar_imagenes

# Definir el modelo CNN
def crear_modelo_cnn(tamaño_imagen=(128, 128)):
    modelo = models.Sequential([
        layers.Input(shape=(tamaño_imagen[0], tamaño_imagen[1], 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Clasificación binaria
    ])
    return modelo

# Cargar y preprocesar las imágenes
data_dir = '../data/'
imágenes = cargar_y_preprocesar_imagenes(data_dir)
imágenes = np.expand_dims(imágenes, axis=-1)  # Añadir canal de dimensión

# Ajustar las etiquetas a la cantidad de imágenes
etiquetas = np.array([0, 1, 0, 1, 0, 1])  # Asegúrate de que haya una etiqueta por imagen

# Verificar la cantidad de imágenes y etiquetas
print(f'Cantidad de imágenes: {len(imágenes)}')
print(f'Cantidad de etiquetas: {len(etiquetas)}')

# Crear y compilar el modelo
modelo = crear_modelo_cnn()
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(imágenes, etiquetas, epochs=5)

# Guardar el modelo
modelo.save('modelo_cnn.h5')
