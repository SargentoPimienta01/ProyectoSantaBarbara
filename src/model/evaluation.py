from keras import models

import numpy as np

# Cargar modelo y datos de validación
model = models.load_model("trained_models/egg_detection_model.h5")
X_val = np.load("data/X_val.npy")
y_val_classes = np.load("data/y_val_classes.npy")

# Evaluar
loss, accuracy = model.evaluate(X_val, y_val_classes)
print(f"Loss: {loss}, Accuracy: {accuracy}")
