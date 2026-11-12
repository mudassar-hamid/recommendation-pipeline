import json
from pathlib import Path
import pandas as pd

from src.validation.validate import validate_data


def test_validator_quarantines_invalid_rows(tmp_path: Path):
    normalized = tmp_path / "raw" / "normalized"
    processed = tmp_path / "processed"
    errors = tmp_path / "errors"
    reports = tmp_path / "reports"
    for folder in (normalized, processed, errors, reports):
        folder.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        {"user_id": 1, "gender": "F", "age_group_code": 25, "occupation_code": 1, "zip_code": "00123"},
        {"user_id": 2, "gender": None, "age_group_code": 35, "occupation_code": 2, "zip_code": "00234"},
    ]).to_csv(normalized / "users.csv", index=False)
    pd.DataFrame([
        {"item_id": 10, "title": "A", "genres": "Drama"},
        {"item_id": 11, "title": "B", "genres": "Comedy"},
    ]).to_csv(normalized / "items.csv", index=False)
    stamp = "2020-01-01T00:00:00+00:00"
    pd.DataFrame([
        {"user_id": 1, "item_id": 10, "rating": 5, "timestamp": stamp},
        {"user_id": 1, "item_id": 10, "rating": 6, "timestamp": "2020-01-02T00:00:00+00:00"},
        {"user_id": 1, "item_id": 999, "rating": 4, "timestamp": "2020-01-03T00:00:00+00:00"},
        {"user_id": 1, "item_id": 11, "rating": 4, "timestamp": "not-a-date"},
        {"user_id": 2, "item_id": 10, "rating": 4, "timestamp": "2020-01-04T00:00:00+00:00"},
        {"user_id": 1, "item_id": 10, "rating": 5, "timestamp": stamp},
        {"user_id": 1, "item_id": 11, "rating": 4, "timestamp": "2020-01-05T00:00:00+00:00"},
    ]).to_csv(normalized / "interactions.csv", index=False)

    config = {"paths": {"normalized_dir": normalized, "processed_dir": processed,
                        "errors_dir": errors, "reports_dir": reports}}
    validate_data(config)

    invalid = pd.read_csv(errors / "invalid_interactions.csv")
    assert len(invalid) == 6
    reasons = ";".join(invalid["validation_errors"].astype(str).tolist())
    assert "rating_outside_1_to_5" in reasons
    assert "unknown_item_reference" in reasons
    assert "invalid_timestamp" in reasons
    assert "duplicate_interaction_event" in reasons
    assert "unknown_user_reference" in reasons

    report = json.loads((reports / "validation_report.json").read_text(encoding="utf-8"))
    assert report["datasets"]["users"]["invalid"] == 1
    assert report["datasets"]["interactions"]["valid"] == 1
    assert len(pd.read_csv(processed / "validated_interactions.csv")) == 1
