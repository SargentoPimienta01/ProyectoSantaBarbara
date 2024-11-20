from preprocessing.load_data import load_images_and_labels
from preprocessing.process_labels import convert_labels_to_bboxes
from sklearn.model_selection import train_test_split
import numpy as np

# Mapeo de etiquetas
label_map = {
    "huevo": 0,
    "fisura": 1,
    "textura_rugosa": 2,
    "viable": 3,
    "no_viable": 4
}

# Directorios
image_dir = "data/images"
annotation_dir = "data/annotations"

# Cargar imágenes y etiquetas
images, raw_labels = load_images_and_labels(image_dir, annotation_dir)

# Convertir etiquetas
bboxes, classes = convert_labels_to_bboxes(raw_labels, label_map)

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    images, (bboxes, classes), test_size=0.2, random_state=42
)

# Guardar los datos procesados
np.save("data/X_train.npy", X_train)
np.save("data/X_val.npy", X_val)
np.save("data/y_train_bboxes.npy", y_train[0])
np.save("data/y_train_classes.npy", y_train[1])
np.save("data/y_val_bboxes.npy", y_val[0])
np.save("data/y_val_classes.npy", y_val[1])
