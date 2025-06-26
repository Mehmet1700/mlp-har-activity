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
import plotly.express as px
from datetime import datetime


# MLflow-Tracking-URI setzen
# MLflow Setup
mlflow.set_tracking_uri("file:logs/mlruns")
mlflow.set_experiment("LSTM_HAR_Optuna")  


X_train, y_train, X_val, y_val, label_encoder = prepare_data("data/train.csv", "data/test.csv")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial, parent_run_id):
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)


    epochs = 100

    # MLflow: Parameter und Metriken loggen
    run_name = f"Trial{trial.number}_h{hidden_size}_b{batch_size}_lr{lr:.0e}"
    with mlflow.start_run(run_name=f"Trial_{trial.number}", run_id=None, nested=True, parent_run_id=parent_run_id) as run:
        mlflow.set_tag("trial_group", parent_run_name)
        mlflow.set_tag("is_parent", "true") 

        

        # train_model gibt jetzt auch das Modell zurück
        train_acc, train_loss, train_prec, train_rec, train_f1, val_acc, val_loss, val_prec, val_rec, val_f1, model  = train_model(
            X_train, y_train, X_val, y_val,
            hidden_size=hidden_size,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            dropout=dropout,
            activation=activation,
            num_layers=num_layers,
            weight_decay=weight_decay,
            device=device,
            log_to_mlflow=True  
        )

        mlflow.log_params({
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
            "activation": activation,
            "run_name": run_name,
            "epochs": epochs,
            "num_layers": num_layers,
            "weight_decay": weight_decay
        })
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "train_precision": train_prec,
            "train_recall": train_rec,
            "train_f1": train_f1,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1
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
    trial.set_user_attr("val_accuracy", val_acc)
    trial.set_user_attr("val_loss", val_loss)
    trial.set_user_attr("val_precision", val_prec)
    trial.set_user_attr("val_recall", val_rec)
    trial.set_user_attr("val_f1", val_f1)
    trial.set_user_attr("mlflow_parent_run_id", parent_run_id)




    print(f"[Trial {trial.number}] val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, precision={val_prec:.4f}, recall={val_rec:.4f}, f1={val_f1:.4f}")


    return val_acc


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    parent_run_name = f"Optuna_Loop_{timestamp}"
    with mlflow.start_run(run_name=parent_run_name, nested=False) as parent_run:
        mlflow.set_tags({
            "is_parent": "true",
            "experiment": "LSTM-HAR",
            "timestamp": timestamp
        })
        study = optuna.create_study(
            direction="maximize",
            study_name="lstm_har_study",
            storage="sqlite:///optuna_lstm.db",
            load_if_exists=True
    )
        study.optimize(lambda trial: objective(trial, parent_run.info.run_id), n_trials=60, n_jobs=1)
        mlflow.log_metrics({
            "best_val_accuracy": study.best_trial.user_attrs.get("val_accuracy", 0),
            "best_val_loss": study.best_trial.user_attrs.get("val_loss", 0),
            "best_val_precision": study.best_trial.user_attrs.get("val_precision", 0),
            "best_val_recall": study.best_trial.user_attrs.get("val_recall", 0),
            "best_val_f1": study.best_trial.user_attrs.get("val_f1", 0)
        })

        # Alle Trials dieses Parent-Runs extrahieren 
        current_trials = [
            t for t in study.trials 
            if t.user_attrs.get("mlflow_parent_run_id") == parent_run.info.run_id
        ]

        if current_trials:
            records = []
            for trial in current_trials:
                trial_id = trial.number
                metrics = trial.user_attrs
                records.append({
                    "trial": f"Trial {trial_id}",
                    "accuracy": metrics.get("val_accuracy", 0),
                    "loss": metrics.get("val_loss", 0),
                    "precision": metrics.get("val_precision", 0),
                    "recall": metrics.get("val_recall", 0),
                    "f1": metrics.get("val_f1", 0)
                })

            df = pd.DataFrame(records)

            os.makedirs("plots", exist_ok=True)

            # Einzeln geplottete Metriken
            metrics_name = ["accuracy", "loss", "precision", "recall", "f1"]
            for metric in metrics_name:
                fig = px.bar(df, x="trial", y=metric, title=f"Trials – {metric.capitalize()}", text_auto='.3s')
                fig.update_layout(xaxis_title="Trial", yaxis_title=metric.capitalize())
                file_name = f"plots/metrics_{metric}.html"
                fig.write_html(file_name)
                mlflow.log_artifact(file_name)

            #  Vergleich aller Metriken
            df_melted = df.melt(id_vars="trial", value_vars=metrics_name, var_name="metric", value_name="value")
            fig = px.bar(df_melted, x="trial", y="value", color="metric", barmode="group",
                        title="Vergleich aller Trials (val-Metriken)")
            fig.update_layout(xaxis_title="Trial", yaxis_title="Value")
            fig.write_html("plots/metrics_comparison.html")
            mlflow.log_artifact("plots/metrics_comparison.html")

            # CSV speichern
            df.to_csv("plots/current_trials_metrics.csv", index=False)
            mlflow.log_artifact("plots/current_trials_metrics.csv")

        else:
            print(" Keine zugehörigen Trials für diesen Parent-Run gefunden.")









        mlflow.end_run()




    print("Beste Parameter:", study.best_params)
    print("Beste Accuracy:", study.best_value)

