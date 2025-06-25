import optuna
import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from train_lstm import train_model
from train_lstm import prepare_data
from optuna.visualization import plot_optimization_history, plot_param_importances
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score
import math
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature


# MLflow-Tracking-URI setzen
# MLflow Setup
mlflow.set_tracking_uri("file:logs/mlruns")
mlflow.set_experiment("LSTM_HAR_Optuna")  


X_train, y_train, X_val, y_val, label_encoder = prepare_data("data/train.csv", "data/test.csv")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    epochs = 100

    # train_model gibt jetzt auch das Modell zurück
    acc, val_loss, precision, recall, model = train_model(
        X_train, y_train, X_val, y_val,
        hidden_size=hidden_size,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        dropout=dropout,
        activation=activation,
        device=device,
        mlflow_logging=True  # MLflow Logging aktivieren
    )

    # MLflow: Parameter und Metriken loggen
    run_name = f"Trial{trial.number}_h{hidden_size}_b{batch_size}_lr{lr:.0e}"
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
            "activation": activation,
            "run_name": run_name,
            "epochs": epochs
        })
        mlflow.log_metrics({
            "val_accuracy": acc,
            "val_loss": val_loss if val_loss is not None else -1.0,
            "precision": precision if precision is not None else 0.0,
            "recall": recall if recall is not None else 0.0
        })

        # MLflow: Modell + Signature + Input Example loggen
        input_tensor = X_val[:1].to(device)
        input_example = input_tensor.cpu().numpy()    

        model.eval()
        with torch.no_grad():
            output_example = model(torch.tensor(input_example).to(device)).cpu().numpy()
        # Signature inferieren
        signature = infer_signature(input_example, output_example)

        # Modell speichern
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )
    # Optuna User Attribute speichern
    trial.set_user_attr("val_accuracy", acc)
    trial.set_user_attr("val_loss", val_loss)
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    



    print(f"[Trial] acc={acc:.4f}, "
        f"loss={'{:.4f}'.format(val_loss) if val_loss is not None else 'NA'}, "
        f"precision={'{:.4f}'.format(precision) if precision is not None else 'NA'}, "
        f"recall={'{:.4f}'.format(recall) if recall is not None else 'NA'}")


    return acc

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="lstm_har_study",
        storage="sqlite:///optuna_lstm.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=100, n_jobs=1)

    print("Beste Parameter:", study.best_params)
    print("Beste Accuracy:", study.best_value)

    # Bestes Modell separat in MLflow loggen
    best_trial = study.best_trial
    with mlflow.start_run(run_name="Best_Model", nested=False):
        mlflow.log_params(best_trial.params)
        # Optuna-User-Attribute sind ein Dictionary:
        for k, v in best_trial.user_attrs.items():
            mlflow.log_metric(k, v)


