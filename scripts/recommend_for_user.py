"""CLI example for generating recommendations from trained model artifacts."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.recommend import recommend_for_user


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend movies for a MovieLens user")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--model", choices=["personalized", "popularity"], default="personalized")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    recs = recommend_for_user(args.user_id, args.config, args.model, args.k)
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
