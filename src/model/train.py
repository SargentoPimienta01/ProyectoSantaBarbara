import numpy as np
from keras import utils
from build import build_model

def train_model(X_train_path, y_train_path, X_val_path, y_val_path, output_model_path):
    """
    Entrena un modelo con datos de entrenamiento y validación.

    Args:
        X_train_path (str): Ruta al archivo de datos de entrenamiento (imágenes).
        y_train_path (str): Ruta al archivo de etiquetas de entrenamiento.
        X_val_path (str): Ruta al archivo de datos de validación (imágenes).
        y_val_path (str): Ruta al archivo de etiquetas de validación.
        output_model_path (str): Ruta para guardar el modelo entrenado.
    """
    # Cargar datos de entrenamiento y validación
    print("Cargando datos...")
    X_train = np.load(X_train_path)
    X_val = np.load(X_val_path)
    y_train_classes = np.load(y_train_path)
    y_val_classes = np.load(y_val_path)

    # Convertir etiquetas a formato one-hot
    print("Preparando etiquetas...")
    y_train_classes = utils.to_categorical(y_train_classes)
    y_val_classes = utils.to_categorical(y_val_classes)

    # Construir el modelo
    print("Construyendo el modelo...")
    model = build_model(input_shape=(224, 224, 3), num_classes=y_train_classes.shape[1])

    # Entrenar el modelo
    print("Entrenando el modelo...")
    model.fit(X_train, y_train_classes, validation_data=(X_val, y_val_classes), epochs=10, batch_size=32)

    # Guardar el modelo entrenado
    print(f"Guardando el modelo en {output_model_path}...")
    model.save(output_model_path)
    print("Entrenamiento completado.")
