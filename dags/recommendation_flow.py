"""Prefect orchestration for the end-to-end MovieLens recommendation pipeline."""
from __future__ import annotations

import argparse
from prefect import flow, task

from src.config import load_config
from src.ingestion.ingest import ingest_data
from src.validation.validate import validate_data
from src.preparation.prepare import prepare_data
from src.transformation.transform import transform_data
from src.features.build_features import build_features
from src.training.train import train_popularity_model, train_personalized_model
from src.evaluation.evaluate import evaluate_models


@task(name="1 - Ingest data", retries=2, retry_delay_seconds=5, log_prints=True)
def ingestion_task(config_path: str) -> dict[str, str]:
    return ingest_data(load_config(config_path))


@task(name="2 - Validate data", retries=2, retry_delay_seconds=5, log_prints=True)
def validation_task(config_path: str) -> dict[str, str]:
    return validate_data(load_config(config_path))


@task(name="3 - Prepare data", retries=2, retry_delay_seconds=5, log_prints=True)
def preparation_task(config_path: str) -> dict[str, str]:
    return prepare_data(load_config(config_path))


@task(name="4 - Transform data", retries=2, retry_delay_seconds=5, log_prints=True)
def transformation_task(config_path: str) -> dict[str, str]:
    return transform_data(load_config(config_path))


@task(name="5 - Build features", retries=2, retry_delay_seconds=5, log_prints=True)
def feature_task(config_path: str) -> dict[str, str]:
    return build_features(load_config(config_path))


@task(name="6 - Initialize MLflow experiment", retries=1, retry_delay_seconds=5, log_prints=True)
def initialize_mlflow_task(config_path: str) -> dict[str, str]:
    config = load_config(config_path)
    if not config.get("mlflow", {}).get("enabled", True):
        return {"mlflow": "disabled"}
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is enabled but missing. Install dependencies with: pip install -r requirements.txt") from exc
    tracking_uri = config["mlflow"]["tracking_uri"]
    print(f"MLflow Tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Actual MLflow URI: {mlflow.get_tracking_uri()}")
    experiment = mlflow.set_experiment(config["mlflow"].get("experiment_name", "MovieLens-Recommendation-Pipeline"))
    print(f"MLflow experiment ready: {experiment.name}")
    return {"experiment_id": experiment.experiment_id}


@task(name="6A - Train popularity baseline", retries=1, retry_delay_seconds=5, log_prints=True)
def popularity_training_task(config_path: str) -> dict[str, str]:
    return train_popularity_model(load_config(config_path))


@task(name="6B - Train personalized model", retries=1, retry_delay_seconds=5, log_prints=True)
def personalized_training_task(config_path: str) -> dict[str, str]:
    return train_personalized_model(load_config(config_path))


@task(name="7 - Evaluate and report", retries=1, retry_delay_seconds=5, log_prints=True)
def evaluation_task(config_path: str) -> dict[str, str]:
    return evaluate_models(load_config(config_path))


@flow(name="MovieLens End-to-End Recommendation Pipeline", log_prints=True)
def recommendation_pipeline(config_path: str = "configs/config.yaml") -> dict[str, str]:
    """Run all stages in dependency order; model training branches after features."""
    # Explicit wait_for edges create a visible dependency graph in Prefect.
    ingested = ingestion_task.submit(config_path)
    validated = validation_task.submit(config_path, wait_for=[ingested])
    prepared = preparation_task.submit(config_path, wait_for=[validated])
    transformed = transformation_task.submit(config_path, wait_for=[prepared])
    featured = feature_task.submit(config_path, wait_for=[transformed])

    # Initialize the shared local experiment before training tasks branch.
    tracking_ready = initialize_mlflow_task.submit(config_path, wait_for=[featured])

    # These model branches are independent after feature generation and tracking setup.
    popularity_model = popularity_training_task.submit(config_path, wait_for=[tracking_ready])
    personalized_model = personalized_training_task.submit(config_path, wait_for=[tracking_ready])
    evaluated = evaluation_task.submit(config_path, wait_for=[popularity_model, personalized_model])
    final_outputs = evaluated.result()
    print("Recommendation pipeline completed successfully.")
    return final_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Prefect recommendation pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML configuration")
    args = parser.parse_args()
    recommendation_pipeline(config_path=args.config)
