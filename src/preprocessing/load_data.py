import os
import json
import tensorflow as tf
import numpy as np

def load_images_and_labels(image_dir, annotation_dir):
    images = []
    labels = []
    for file in os.listdir(annotation_dir):
        if file.endswith(".json"):
            # Leer anotaciones
            with open(os.path.join(annotation_dir, file), "r") as f:
                annotation = json.load(f)

            image_path = os.path.join(image_dir, annotation["imagePath"])
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            image = tf.keras.preprocessing.image.img_to_array(image)

            labels.append(annotation["shapes"])
            images.append(image)

    return np.array(images), labels
