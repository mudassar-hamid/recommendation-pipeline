"""Download MovieLens 1M and normalize the source files into CSV datasets."""
from __future__ import annotations

from pathlib import Path
import urllib.request
import zipfile
import pandas as pd

from src.common.io import write_csv, write_json


def _download_and_extract(url: str, raw_dir: Path, source_dir: Path) -> None:
    required = [source_dir / name for name in ("users.dat", "movies.dat", "ratings.dat")]
    if all(path.exists() for path in required):
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "ml-1m.zip"
    if not archive.exists():
        print(f"Downloading MovieLens 1M from {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "DML-Assignment-Recommendation-Pipeline/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
            output.write(response.read())

    with zipfile.ZipFile(archive, "r") as zf:
        # Extract only the expected MovieLens directory and files.
        expected = {"ml-1m/users.dat", "ml-1m/movies.dat", "ml-1m/ratings.dat", "ml-1m/README"}
        members = [name for name in zf.namelist() if name in expected]
        if not {"ml-1m/users.dat", "ml-1m/movies.dat", "ml-1m/ratings.dat"}.issubset(set(members)):
            raise ValueError("Downloaded archive does not contain the expected MovieLens 1M files")
        zf.extractall(raw_dir, members=members)

    if not all(path.exists() for path in required):
        raise FileNotFoundError("MovieLens extraction did not produce users.dat, movies.dat and ratings.dat")


def ingest_data(config: dict) -> dict[str, str]:
    paths = config["paths"]
    normalized_dir: Path = paths["normalized_dir"]
    normalized_files = [normalized_dir / name for name in ("users.csv", "items.csv", "interactions.csv")]

    # This makes local reruns and synthetic smoke tests possible without redownloading data.
    if all(path.exists() for path in normalized_files):
        print("Normalized source CSVs already exist; reusing them.")
        return {"users": str(normalized_files[0]), "items": str(normalized_files[1]), "interactions": str(normalized_files[2])}

    _download_and_extract(config["data"]["dataset_url"], paths["raw_dir"], paths["source_dir"])
    source_dir: Path = paths["source_dir"]

    users = pd.read_csv(
        source_dir / "users.dat", sep=r"::", engine="python", encoding="latin-1",
        names=["user_id", "gender", "age_group_code", "occupation_code", "zip_code"],
    )
    items = pd.read_csv(
        source_dir / "movies.dat", sep=r"::", engine="python", encoding="latin-1",
        names=["item_id", "title", "genres"],
    )
    interactions = pd.read_csv(
        source_dir / "ratings.dat", sep=r"::", engine="python", encoding="latin-1",
        names=["user_id", "item_id", "rating", "timestamp_unix"],
    )

    users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce").astype("Int64")
    users["age_group_code"] = pd.to_numeric(users["age_group_code"], errors="coerce").astype("Int64")
    users["occupation_code"] = pd.to_numeric(users["occupation_code"], errors="coerce").astype("Int64")
    users["zip_code"] = users["zip_code"].astype("string")

    items["item_id"] = pd.to_numeric(items["item_id"], errors="coerce").astype("Int64")
    items["title"] = items["title"].astype("string")
    items["genres"] = items["genres"].astype("string")

    interactions["user_id"] = pd.to_numeric(interactions["user_id"], errors="coerce").astype("Int64")
    interactions["item_id"] = pd.to_numeric(interactions["item_id"], errors="coerce").astype("Int64")
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions["timestamp"] = pd.to_datetime(
        pd.to_numeric(interactions.pop("timestamp_unix"), errors="coerce"),
        unit="s", utc=True, errors="coerce",
    )

    normalized_dir.mkdir(parents=True, exist_ok=True)
    user_path = write_csv(users, normalized_files[0])
    item_path = write_csv(items, normalized_files[1])
    interaction_path = write_csv(interactions, normalized_files[2])

    metadata = {
        "dataset": "MovieLens 1M",
        "source_url": config["data"]["dataset_url"],
        "source_format": "MovieLens DAT files using :: delimiter; normalized to CSV",
        "users_rows": int(len(users)),
        "items_rows": int(len(items)),
        "interactions_rows": int(len(interactions)),
        "note": "MovieLens age_group_code values are bucket codes, not exact ages.",
    }
    write_json(metadata, paths["raw_dir"] / "ingestion_metadata.json")
    print(f"Ingested users={len(users):,}, items={len(items):,}, ratings={len(interactions):,}")
    return {"users": str(user_path), "items": str(item_path), "interactions": str(interaction_path)}
