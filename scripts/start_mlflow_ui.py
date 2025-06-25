import os
import subprocess

# === Konfiguration ===
mlruns_dir = "logs/mlruns"
mlflow_port = 5000

# === Schritt 1: Sicherstellen, dass logs/mlruns existiert ===
os.makedirs(mlruns_dir, exist_ok=True)

# === Schritt 2: MLflow UI starten ===
print(f"[INFO] Starte MLflow UI auf Port {mlflow_port} ...")
print(f"[INFO] Backend-Store-URI: file:{mlruns_dir}")

try:
    subprocess.run([
        "mlflow", "ui",
        "--backend-store-uri", f"file:{mlruns_dir}",
        "--port", str(mlflow_port)
    ])
except FileNotFoundError:
    print("[ERROR] MLflow ist nicht installiert oder nicht im PATH. Bitte installiere es mit: pip install mlflow")
