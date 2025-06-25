import mlflow
import pandas as pd
import streamlit as st
import plotly.express as px

# MLflow Tracking-URI setzen (lokaler Pfad)
mlflow.set_tracking_uri("file:logs/mlruns")

# Dein Experiment auswählen
experiment_name = "LSTM_HAR_Optuna"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    st.error(f"Experiment '{experiment_name}' not found.")
    st.stop()

# Läufe holen (inkl. Kindläufe)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    output_format="pandas"
)

# Nur Child-Runs (nested)
child_runs = runs[runs['tags.mlflow.parentRunId'].notnull()].copy()
child_runs['trial'] = child_runs['tags.mlflow.runName']

st.title("Optuna LSTM Dashboard")
st.write(f"Experiment: `{experiment_name}` – {len(child_runs)} Runs gefunden")

# Auswahl der Metrik für Top-Trials
metric = st.selectbox("Wähle Metrik zum Sortieren", ["metrics.val_accuracy", "metrics.f1", "metrics.val_loss", "metrics.precision", "metrics.recall"])

top_n = st.slider("Anzahl Top Trials", 5, 50, 20)

# Sortierte Top-Runs
sorted_runs = child_runs.sort_values(by=metric, ascending="loss" in metric).head(top_n)

# Balkendiagramm
fig = px.bar(
    sorted_runs,
    x="trial",
    y=metric,
    color="trial",
    title=f"Top {top_n} Trials nach {metric}",
    text=metric
)
fig.update_layout(showlegend=False, xaxis_title="Trial", yaxis_title=metric.split(".")[-1])
st.plotly_chart(fig, use_container_width=True)

# Alle Metriken auf einmal anzeigen
st.subheader("Vergleich der Metriken (Top Trials)")

melted = sorted_runs.melt(id_vars="trial", value_vars=[
    "metrics.val_accuracy", "metrics.f1", "metrics.precision", "metrics.recall", "metrics.val_loss"
], var_name="Metrik", value_name="Wert")

fig2 = px.bar(
    melted,
    x="trial",
    y="Wert",
    color="Metrik",
    barmode="group",
    title="Vergleich aller Metriken"
)
st.plotly_chart(fig2, use_container_width=True)

# Details anzeigen
st.subheader("Run Details")
selected_trial = st.selectbox("Wähle einen Trial zur Detailansicht", sorted_runs["trial"])
selected_run = sorted_runs[sorted_runs["trial"] == selected_trial].iloc[0]

st.json(selected_run.to_dict())
