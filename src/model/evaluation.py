from keras import models
import numpy as np

def evaluate_model(model_path, X_val_path, y_val_path):
    """
    Evalúa un modelo entrenado con datos de validación.
    
    Args:
        model_path (str): Ruta al archivo del modelo entrenado (.h5).
        X_val_path (str): Ruta a los datos de validación (imágenes).
        y_val_path (str): Ruta a las etiquetas de validación.
    
    Returns:
        dict: Resultados de pérdida y precisión.
    """
    model = models.load_model(model_path)
    print(f"Modelo cargado desde {model_path}")

    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)
    print(f"Datos de validación cargados desde {X_val_path} y {y_val_path}")

    loss, accuracy = models.Model.evaluate(X_val, y_val, verbose=1)
    print(f"Evaluación completa. Pérdida: {loss}, Precisión: {accuracy}")

    return {"loss": loss, "accuracy": accuracy}

