"""Run a tiny offline end-to-end smoke test without Prefect, MLflow or MovieLens download."""
from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.validate import validate_data
from src.preparation.prepare import prepare_data
from src.transformation.transform import transform_data
from src.features.build_features import build_features
from src.training.train import train_popularity_model, train_personalized_model
from src.evaluation.evaluate import evaluate_models
from src.inference.recommend import recommend_user


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dml_rec_smoke_") as temp:
        root = Path(temp)
        paths = {name: root / folder for name, folder in {
            "raw_dir": "raw", "source_dir": "raw/ml-1m", "normalized_dir": "raw/normalized",
            "processed_dir": "processed", "features_dir": "features", "errors_dir": "errors",
            "models_dir": "models", "reports_dir": "reports", "tracking_dir": "mlruns",
        }.items()}
        for path in paths.values(): path.mkdir(parents=True, exist_ok=True)

        users = pd.DataFrame({
            "user_id": [1, 2, 3, 4], "gender": ["F", "M", "F", "M"],
            "age_group_code": [25, 35, 18, 45], "occupation_code": [1, 2, 3, 4],
            "zip_code": ["A1", "B2", "C3", "D4"],
        })
        items = pd.DataFrame({
            "item_id": list(range(1, 9)),
            "title": [f"Movie {i}" for i in range(1, 9)],
            "genres": ["Action|Drama", "Comedy", "Drama", "Action", "Romance", "Comedy|Drama", "Thriller", "Adventure"],
        })
        events = []
        for day in range(1, 7):
            for user_id in range(1, 5):
                item_id = ((day - 1) + 2 * user_id) % 8 + 1
                rating = 5.0 if day == 6 or (user_id + item_id) % 2 == 0 else 3.0
                events.append({"user_id": user_id, "item_id": item_id, "rating": rating,
                               "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=day)})
        interactions = pd.DataFrame(events)
        normalized = paths["normalized_dir"]
        users.to_csv(normalized / "users.csv", index=False)
        items.to_csv(normalized / "items.csv", index=False)
        interactions.to_csv(normalized / "interactions.csv", index=False)

        config = {
            "paths": paths,
            "data": {"train_fraction": 0.80, "positive_rating_threshold": 4.0},
            "evaluation": {"k": 3, "item_similarity_neighbors": 5},
            "mlflow": {"enabled": False},
            "report": {"generate_pdf": True},
        }
        validate_data(config)
        prepare_data(config)
        transform_data(config)
        build_features(config)
        train_popularity_model(config)
        train_personalized_model(config)
        result = evaluate_models(config)
        comparison = pd.read_csv(result["comparison"])
        assert len(comparison) == 2
        assert {"precision_at_3", "recall_at_3", "ndcg_at_3", "coverage", "diversity"}.issubset(comparison.columns)
        assert (Path(result["comparison"]).parent / "final_report.pdf").exists()
        individual = recommend_user(1, config, model_type="personalized", k=3)
        assert len(individual) <= 3 and "recommendation_source" in individual.columns
        new_user = recommend_user(999, config, model_type="personalized", k=3)
        assert "popularity fallback" in set(new_user["recommendation_source"])
        print("OFFLINE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
