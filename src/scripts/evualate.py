from model.evaluation import evaluate_model

if __name__ == "__main__":
    model_path = "trained_models/egg_detection_model.h5"
    X_val_path = "data/processed/X_val.npy"
    y_val_path = "data/processed/y_val.npy"

    results = evaluate_model(model_path, X_val_path, y_val_path)

    print("Resultados de la evaluación:")
    print(f"Pérdida: {results['loss']}")
    print(f"Precisión: {results['accuracy']}")
