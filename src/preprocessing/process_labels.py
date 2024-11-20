import numpy as np

def convert_labels_to_bboxes(labels, label_map):
    bboxes = []
    classes = []
    for label in labels:
        for obj in label:
            x_min = min(point[0] for point in obj["points"])
            y_min = min(point[1] for point in obj["points"])
            x_max = max(point[0] for point in obj["points"])
            y_max = max(point[1] for point in obj["points"])

            bboxes.append([x_min, y_min, x_max, y_max])
            classes.append(label_map[obj["label"]])

    return np.array(bboxes), np.array(classes)
