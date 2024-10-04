import tensorflow as tf
from keras import layers, models
import numpy as np
from cargar_imagenes import cargar_y_preprocesar_imagenes, colorimetria, detectar_huevos

# Definir el modelo CNN
def crear_modelo_cnn(tamaño_imagen=(128, 128)):
    modelo = models.Sequential([
        layers.Input(shape=(tamaño_imagen[0], tamaño_imagen[1], 1)),  # Imagen en escala de grises (1 canal)
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Clasificación binaria (viable o no viable)
    ])
    return modelo

# Directorio donde están almacenadas las imágenes
data_dir = 'data'

# Detección y preprocesamiento de los huevos en las imágenes
huevos_detectados = detectar_huevos(data_dir)
if len(huevos_detectados) == 0:
    raise Exception("No se detectaron huevos en las imágenes.")

# Colorimetría (opcional, dependiendo de si quieres usar este análisis)
# Aquí podrías implementar más procesamiento si es necesario.

# Preprocesar las imágenes (convertir a escala de grises y normalizar)
imágenes = cargar_y_preprocesar_imagenes(data_dir)

# Verificar si se han preprocesado correctamente las imágenes
if len(imágenes) == 0:
    raise Exception("No se encontraron imágenes para preprocesar.")

# Asegurarse de que las imágenes tengan el canal necesario para la red neuronal
imágenes = np.expand_dims(imágenes, axis=-1)  # Añadir canal de dimensión para cumplir con el input del modelo

# Ajustar las etiquetas a la cantidad de imágenes (esto debe adaptarse a tus datos reales)
# En este ejemplo estamos usando una cantidad fija de etiquetas para un conjunto de datos de entrenamiento simple
# Si tienes un dataset más grande o con más clases, las etiquetas deben corresponder correctamente
etiquetas = np.array([0, 1] * (len(imágenes) // 2))  # Alterna entre viable (1) y no viable (0)

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

print("Modelo entrenado y guardado correctamente.")
