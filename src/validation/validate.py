"""Automated data-quality checks with row-level error outputs."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import pandas as pd

from src.common.io import read_csv, write_csv, write_json


def _with_reasons(df: pd.DataFrame, reasons: list[list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    invalid_mask = pd.Series([bool(value) for value in reasons], index=df.index)
    invalid = df.loc[invalid_mask].copy()
    if not invalid.empty:
        invalid["validation_errors"] = ["; ".join(reasons[i]) for i in invalid.index]
    valid = df.loc[~invalid_mask].copy()
    return valid, invalid



def _inject_validation_test_records(
    interactions: pd.DataFrame, settings: dict
) -> tuple[pd.DataFrame, set[int], dict]:
    """Append deliberately invalid synthetic rows for validation demonstrations.

    The source CSV is never modified. Injected rows exist only in this in-memory
    validation batch and are quarantined by the normal checks below.
    """
    if not settings.get("enabled", False):
        return interactions, set(), {"enabled": False, "injected_records": 0, "rules": []}

    count = int(settings.get("number_of_records", 5))
    rules = settings.get("rules", ["invalid_rating"])
    if count < 1:
        raise ValueError("validation.test_injection.number_of_records must be >= 1")
    if interactions.empty:
        raise ValueError("Cannot inject validation test records: interactions.csv is empty")
    if not isinstance(rules, list) or not rules:
        raise ValueError("validation.test_injection.rules must be a non-empty list")

    supported = {
        "invalid_rating", "missing_user_id", "missing_item_id",
        "invalid_timestamp", "future_timestamp",
    }
    unknown = sorted(set(rules) - supported)
    if unknown:
        raise ValueError(f"Unsupported validation test injection rules: {unknown}")

    samples = interactions.iloc[[i % len(interactions) for i in range(count)]].copy().reset_index(drop=True)
    applied_rules = []
    for i, (idx, rule) in enumerate(zip(samples.index, [rules[j % len(rules)] for j in range(count)])):
        if rule == "invalid_rating":
            samples["rating"] = samples["rating"].astype(object)
            samples.at[idx, "rating"] = 99
        elif rule == "missing_user_id":
            samples["user_id"] = samples["user_id"].astype(object)
            samples.at[idx, "user_id"] = pd.NA
        elif rule == "missing_item_id":
            samples["item_id"] = samples["item_id"].astype(object)
            samples.at[idx, "item_id"] = pd.NA
        elif rule == "invalid_timestamp":
            samples["timestamp"] = samples["timestamp"].astype(object)
            samples.at[idx, "timestamp"] = "not-a-valid-timestamp"
        elif rule == "future_timestamp":
            samples["timestamp"] = samples["timestamp"].astype(object)
            samples.at[idx, "timestamp"] = "2999-01-01"
        applied_rules.append(rule)

    # Reset indices so the injected records have unique, traceable row positions.
    original = interactions.reset_index(drop=True)
    injected = samples.reset_index(drop=True)
    start = len(original)
    combined = pd.concat([original, injected], ignore_index=True)
    injected_positions = set(range(start, start + len(injected)))
    return combined, injected_positions, {
        "enabled": True,
        "injected_records": int(len(injected)),
        "rules": applied_rules,
        "note": "Synthetic rows were appended in memory only; normalized source files were not modified.",
    }


def validate_data(config: dict) -> dict[str, str]:
    paths = config["paths"]
    raw = paths["normalized_dir"]
    users = read_csv(raw / "users.csv", dtype={"gender": "string", "zip_code": "string"})
    items = read_csv(raw / "items.csv", dtype={"title": "string", "genres": "string"})
    interactions = read_csv(raw / "interactions.csv")
    injection_settings = config.get("validation", {}).get("test_injection", {})
    interactions, injected_positions, injection_report = _inject_validation_test_records(
        interactions, injection_settings
    )
    checks: dict[str, dict] = {}

    # Users: primary key, required profile fields and expected numeric ID.
    u = users.copy()
    u["user_id_num"] = pd.to_numeric(u["user_id"], errors="coerce")
    u["age_group_num"] = pd.to_numeric(u.get("age_group_code"), errors="coerce")
    u["occupation_num"] = pd.to_numeric(u.get("occupation_code"), errors="coerce")
    u_reasons: list[list[str]] = [[] for _ in range(len(u))]
    for pos, (_, row) in enumerate(u.iterrows()):
        if pd.isna(row["user_id_num"]):
            u_reasons[pos].append("missing_or_non_numeric_user_id")
        elif float(row["user_id_num"]) <= 0 or not float(row["user_id_num"]).is_integer():
            u_reasons[pos].append("user_id_must_be_positive_integer")
        for col in [c for c in users.columns if c != "user_id"]:
            if pd.isna(row[col]) or str(row[col]).strip() == "":
                u_reasons[pos].append(f"missing_{col}")
        if "gender" in users.columns and not pd.isna(row.get("gender")) and str(row.get("gender")).upper() not in {"F", "M"}:
            u_reasons[pos].append("invalid_gender_code")
        if "age_group_code" in users.columns:
            age_code = row.get("age_group_num")
            if pd.isna(age_code):
                u_reasons[pos].append("age_group_code_must_be_numeric")
            elif float(age_code) not in {1, 18, 25, 35, 45, 50, 56}:
                u_reasons[pos].append("invalid_age_group_code")
        if "occupation_code" in users.columns:
            occupation_code = row.get("occupation_num")
            if pd.isna(occupation_code) or not float(occupation_code).is_integer() or not 0 <= int(occupation_code) <= 20:
                u_reasons[pos].append("invalid_occupation_code")
    dup_u = u["user_id_num"].duplicated(keep=False) & u["user_id_num"].notna()
    for pos in dup_u[dup_u].index:
        u_reasons[u.index.get_loc(pos)].append("duplicate_user_id")
    valid_users, invalid_users = _with_reasons(users, u_reasons)
    valid_user_ids = set(pd.to_numeric(valid_users["user_id"], errors="coerce").dropna().astype(int))
    checks["users"] = {"total": int(len(users)), "valid": int(len(valid_users)), "invalid": int(len(invalid_users)),
                       "issues": dict(Counter(reason for row in u_reasons for reason in row))}

    # Items: primary key, required title and genres.
    it = items.copy()
    it["item_id_num"] = pd.to_numeric(it["item_id"], errors="coerce")
    i_reasons: list[list[str]] = [[] for _ in range(len(it))]
    for pos, (_, row) in enumerate(it.iterrows()):
        if pd.isna(row["item_id_num"]):
            i_reasons[pos].append("missing_or_non_numeric_item_id")
        elif float(row["item_id_num"]) <= 0 or not float(row["item_id_num"]).is_integer():
            i_reasons[pos].append("item_id_must_be_positive_integer")
        for col in [c for c in items.columns if c != "item_id"]:
            if pd.isna(row[col]) or str(row[col]).strip() == "":
                i_reasons[pos].append(f"missing_{col}")
    dup_i = it["item_id_num"].duplicated(keep=False) & it["item_id_num"].notna()
    for pos in dup_i[dup_i].index:
        i_reasons[it.index.get_loc(pos)].append("duplicate_item_id")
    valid_items, invalid_items = _with_reasons(items, i_reasons)
    valid_item_ids = set(pd.to_numeric(valid_items["item_id"], errors="coerce").dropna().astype(int))
    checks["items"] = {"total": int(len(items)), "valid": int(len(valid_items)), "invalid": int(len(invalid_items)),
                       "issues": dict(Counter(reason for row in i_reasons for reason in row))}

    # Interactions: key fields, numeric rating in [1, 5], parseable non-future timestamp,
    # duplicate event key and foreign-key checks against valid entity IDs.
    x = interactions.copy()
    x["user_id_num"] = pd.to_numeric(x["user_id"], errors="coerce")
    x["item_id_num"] = pd.to_numeric(x["item_id"], errors="coerce")
    x["rating_num"] = pd.to_numeric(x["rating"], errors="coerce")
    x["timestamp_parsed"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    x_reasons: list[list[str]] = [[] for _ in range(len(x))]
    for pos, (_, row) in enumerate(x.iterrows()):
        if pd.isna(row["user_id_num"]):
            x_reasons[pos].append("missing_or_non_numeric_user_id")
        elif float(row["user_id_num"]) <= 0 or not float(row["user_id_num"]).is_integer():
            x_reasons[pos].append("user_id_must_be_positive_integer")
        if pd.isna(row["item_id_num"]):
            x_reasons[pos].append("missing_or_non_numeric_item_id")
        elif float(row["item_id_num"]) <= 0 or not float(row["item_id_num"]).is_integer():
            x_reasons[pos].append("item_id_must_be_positive_integer")
        if pd.isna(row["rating_num"]):
            x_reasons[pos].append("missing_or_non_numeric_rating")
        elif not 1.0 <= float(row["rating_num"]) <= 5.0:
            x_reasons[pos].append("rating_outside_1_to_5")
        elif not float(row["rating_num"]).is_integer():
            x_reasons[pos].append("rating_must_be_whole_star_for_movielens_1m")
        if pd.isna(row["timestamp_parsed"]):
            x_reasons[pos].append("invalid_timestamp")
        elif row["timestamp_parsed"] > pd.Timestamp.now(tz="UTC"):
            x_reasons[pos].append("future_timestamp")
        if (not pd.isna(row["user_id_num"]) and float(row["user_id_num"]).is_integer()
                and int(row["user_id_num"]) not in valid_user_ids):
            x_reasons[pos].append("unknown_user_reference")
        if (not pd.isna(row["item_id_num"]) and float(row["item_id_num"]).is_integer()
                and int(row["item_id_num"]) not in valid_item_ids):
            x_reasons[pos].append("unknown_item_reference")

    event_key = pd.DataFrame({
        "user_id": x["user_id_num"], "item_id": x["item_id_num"], "timestamp": x["timestamp_parsed"]
    })
    dup_x = event_key.duplicated(keep=False) & event_key.notna().all(axis=1)
    for pos in dup_x[dup_x].index:
        x_reasons[x.index.get_loc(pos)].append("duplicate_interaction_event")

    valid_interactions, invalid_interactions = _with_reasons(interactions, x_reasons)
    if not invalid_interactions.empty:
        invalid_interactions["validation_test_injected"] = [
            int(index) in injected_positions for index in invalid_interactions.index
        ]
    if not valid_interactions.empty:
        valid_interactions["user_id"] = pd.to_numeric(valid_interactions["user_id"]).astype(int)
        valid_interactions["item_id"] = pd.to_numeric(valid_interactions["item_id"]).astype(int)
        valid_interactions["rating"] = pd.to_numeric(valid_interactions["rating"]).astype(float)
        valid_interactions["timestamp"] = pd.to_datetime(valid_interactions["timestamp"], utc=True)
    checks["interactions"] = {
        "total": int(len(interactions)), "valid": int(len(valid_interactions)), "invalid": int(len(invalid_interactions)),
        "issues": dict(Counter(reason for row in x_reasons for reason in row)),
    }

    write_csv(valid_users, paths["processed_dir"] / "validated_users.csv")
    write_csv(valid_items, paths["processed_dir"] / "validated_items.csv")
    write_csv(valid_interactions, paths["processed_dir"] / "validated_interactions.csv")
    write_csv(invalid_users, paths["errors_dir"] / "invalid_users.csv")
    write_csv(invalid_items, paths["errors_dir"] / "invalid_items.csv")
    write_csv(invalid_interactions, paths["errors_dir"] / "invalid_interactions.csv")

    total = sum(x["total"] for x in checks.values())
    valid = sum(x["valid"] for x in checks.values())
    report = {
        "status": "passed" if valid == total else "completed_with_invalid_records",
        "datasets": checks,
        "overall": {"records_checked": total, "valid_records": valid, "invalid_records": total - valid},
        "validation_test_injection": injection_report,
        "validation_policy": "Invalid rows are quarantined in data/errors; valid rows continue to preparation.",
    }
    write_json(report, paths["reports_dir"] / "validation_report.json")
    print(f"Validation completed: {valid:,}/{total:,} valid records; {total-valid:,} quarantined.")
    return {"report": str(paths["reports_dir"] / "validation_report.json")}
