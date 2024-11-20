from build import build_model
import numpy as np
from keras import utils

X_train = np.load("data/X_train.npy")
X_val = np.load("data/X_val.npy")
y_train_classes = np.load("data/y_train_classes.npy")
y_val_classes = np.load("data/y_val_classes.npy")

# Convertir etiquetas a formato one-hot
y_train_classes = utils.to_categorical(y_train_classes)
y_val_classes = utils.to_categorical(y_val_classes)

model = build_model(input_shape=(224, 224, 3), num_classes=5)

model.fit(X_train, y_train_classes, validation_data=(X_val, y_val_classes), epochs=10, batch_size=32)

# Guardar el modelo
model.save("trained_models/egg_detection_model.h5")
