from model.train import train_model

if __name__ == "__main__":
    X_train_path = "data/processed/X_train.npy"
    y_train_path = "data/processed/y_train_classes.npy"
    X_val_path = "data/processed/X_val.npy"
    y_val_path = "data/processed/y_val_classes.npy"
    output_model_path = "trained_models/egg_detection_model.h5"

    train_model(X_train_path, y_train_path, X_val_path, y_val_path, output_model_path)
