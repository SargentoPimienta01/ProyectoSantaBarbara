import os
import subprocess

json_dir = "D:\TESIS\Data\PrimerModelo"

# Convertir cada archivo .json
for file in os.listdir(json_dir):
    if file.endswith(".json"):
        subprocess.run(["labelme_json_to_dataset", os.path.join(json_dir, file)])
