# Human Activity Recognition mit LSTM, Optuna und MLflow

Dieses Projekt befasst sich mit der automatisierten Erkennung menschlicher Aktivitäten auf Basis von Zeitreihendaten. Ziel ist es, ein leistungsfähiges LSTM-Modell zu entwickeln, das durch systematisches Hyperparameter-Tuning mit Optuna sowie experimentelles Tracking via MLflow optimiert wird.

## Projektstruktur

```

mlp-har-activity/
│
├── data/                      # CSV-Daten für Training und Test
│   ├── train.csv
│   └── test.csv
│
├── notebooks/                 # Explorative Analysen und Modellaufbau
│   ├── 01\_data\_understanding.ipynb
│   ├── 02\_data\_preparation.ipynb
│   └── 03\_model\_training.ipynb
│
├── scripts/                   # Python-Skripte zur Modellentwicklung
│   ├── train\_lstm.py
│   ├── optuna\_lstm.py
│   ├── mlflow\_dashboard.py
│   └── start\_mlflow\_ui.py
│
├── plots/                     # Visualisierungen und Diagramme
│
├── logs/mlruns/               # MLflow-Logging-Verzeichnis
│
├── mlflow\.db                  # SQLite-Datenbank für MLflow
├── optuna\_lstm.db             # SQLite-Datenbank für Optuna
│
├── Pipfile                    # Pipenv-Konfiguration
├── Pipfile.lock
├── environment.yml            # Alternative Conda-Umgebung
├── .gitignore
└── README.md

````

## Einrichtung

```bash
pipenv install
pipenv shell
````

## Anwendung

### Modelltraining und Hyperparameteroptimierung

```bash
python scripts/optuna_lstm.py
```

### MLflow-Dashboard starten

```bash
python scripts/start_mlflow_ui.py
```

Alternativ manuell:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Anschließend im Browser aufrufen unter: [http://localhost:5000](http://localhost:5000)

## Modell

Das eingesetzte Modell basiert auf einem LSTM (Long Short-Term Memory), das für sequentielle Daten geeignet ist. Es sind verschiedene Modellkonfigurationen möglich:

* Anzahl der LSTM-Schichten (num\_layers)
* Dropout zur Regularisierung
* Aktivierungsfunktionen (ReLU, Tanh)
* Lernrate (learning rate), Batch-Größe und weitere Hyperparameter

## Logging und Auswertung

Das Experiment-Tracking erfolgt über MLflow:

* Metriken pro Epoche: train\_loss, val\_loss, val\_accuracy, precision, recall, f1-score
* Modellartefakte inkl. Input-Signatur werden automatisch gespeichert
* Übersicht aller Durchläufe (Child-Runs) innerhalb eines Hauptdurchlaufs (Parent-Run)
* Optuna-Datenbank speichert alle getesteten Parameterkombinationen

## Ziel des Projekts

* Entwicklung eines robusten HAR-Modells mit reproduzierbarer Optimierung
* Strukturierte Durchführung und Dokumentation der Experimente
* Grundlage für die Ausarbeitung eines wissenschaftlichen Berichts im Stil eines arXiv-Papers

## Autor

Mehmet Karaca
Wirtschaftsinformatik - Data Science
MLP-HAR-Projekt auf dem bwHPC-Cluster