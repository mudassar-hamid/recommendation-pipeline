"""Clean validated records and create a chronological train/test split."""
from __future__ import annotations

import pandas as pd
from src.common.io import read_csv, write_csv, write_json


def prepare_data(config: dict) -> dict[str, str]:
    paths = config["paths"]
    users = read_csv(paths["processed_dir"] / "validated_users.csv")
    items = read_csv(paths["processed_dir"] / "validated_items.csv")
    interactions = read_csv(paths["processed_dir"] / "validated_interactions.csv")

    # Defensive cleaning: the validation stage should already have quarantined these cases.
    users = users.drop_duplicates(subset=["user_id"]).copy()
    items = items.drop_duplicates(subset=["item_id"]).copy()
    interactions = interactions.drop_duplicates(subset=["user_id", "item_id", "timestamp"]).copy()

    for column, fallback in {"gender": "Unknown", "age_group_code": -1,
                             "occupation_code": -1, "zip_code": "Unknown"}.items():
        if column in users.columns:
            users[column] = users[column].fillna(fallback)
    for column in ("title", "genres"):
        if column in items.columns:
            items[column] = items[column].fillna("Unknown")

    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], utc=True, errors="coerce")
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions = interactions.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    interactions = interactions.sort_values(["timestamp", "user_id", "item_id"], kind="mergesort").reset_index(drop=True)

    train_fraction = float(config["data"].get("train_fraction", 0.80))
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("data.train_fraction must be between 0.5 (inclusive) and 1.0 (exclusive)")
    if len(interactions) < 2:
        raise ValueError("At least two valid interactions are required for a train/test split")

    split_index = max(0, min(len(interactions) - 2, int(len(interactions) * train_fraction) - 1))
    cutoff = interactions.loc[split_index, "timestamp"]
    train = interactions.loc[interactions["timestamp"] <= cutoff].copy()
    test = interactions.loc[interactions["timestamp"] > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition; check timestamps or train_fraction")

    write_csv(users, paths["processed_dir"] / "users_clean.csv")
    write_csv(items, paths["processed_dir"] / "items_clean.csv")
    write_csv(train, paths["processed_dir"] / "train_interactions.csv")
    write_csv(test, paths["processed_dir"] / "test_interactions.csv")

    split_metadata = {
        "method": "global chronological split",
        "train_fraction_requested": train_fraction,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "actual_train_fraction": float(len(train) / len(interactions)),
        "cutoff_timestamp_utc": cutoff.isoformat(),
        "train_max_timestamp_utc": train["timestamp"].max().isoformat(),
        "test_min_timestamp_utc": test["timestamp"].min().isoformat(),
        "note": "Rows at the cutoff timestamp are assigned to train; the actual ratio can differ slightly when timestamps tie.",
    }
    write_json(split_metadata, paths["reports_dir"] / "split_report.json")
    print(f"Prepared {len(train):,} train and {len(test):,} test interactions; cutoff={cutoff}.")
    return {"train": str(paths["processed_dir"] / "train_interactions.csv"),
            "test": str(paths["processed_dir"] / "test_interactions.csv")}
