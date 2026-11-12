"""Generate a concise two-page PDF and Markdown summary after evaluation."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.io import read_json


def build_final_report(config: dict, summary: dict | None = None) -> dict[str, str]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    except ImportError as exc:
        raise RuntimeError("PDF report generation requires reportlab. Install requirements.txt") from exc

    paths = config["paths"]
    reports_dir: Path = paths["reports_dir"]
    summary = summary or read_json(reports_dir / "evaluation_summary.json")
    validation = read_json(reports_dir / "validation_report.json")
    split = read_json(reports_dir / "split_report.json")
    comparison = pd.read_csv(reports_dir / "model_comparison.csv")
    k = int(summary["k"])

    pdf_path = reports_dir / "final_report.pdf"
    md_path = reports_dir / "final_report.md"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=19, leading=23, spaceAfter=8))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=7, spaceAfter=5))
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=8.7, leading=12, spaceAfter=4))
    styles.add(ParagraphStyle(name="SmallCell", parent=styles["BodyText"], fontSize=8, leading=10))

    def para(text: str, style="SmallBody"):
        return Paragraph(text, styles[style])

    story = [
        para("End-to-End Data Management Pipeline for a Recommendation System", "ReportTitle"),
        para("DML Assignment 2 | MovieLens 1M | Implementation and evaluation report", "SmallBody"),
        para("1. Objective and dataset", "Section"),
        para(
            "This project implements a reproducible movie recommendation pipeline covering data ingestion, "
            "automated validation, chronological preparation, feature generation, two recommendation models, "
            "ranking evaluation, experiment tracking and Prefect orchestration. The source dataset is MovieLens 1M "
            "from GroupLens. It contains user demographic fields, movie titles/genres and timestamped ratings. "
            "Age is retained as the dataset's age-group code, not interpreted as an exact age."
        ),
        para("2. Pipeline design", "Section"),
        para(
            "The workflow downloads and normalizes the original MovieLens DAT files into three CSV entities: "
            "users, items and interactions. Validation checks missing values, duplicate entity/event keys, "
            "numeric types, rating range (1–5), timestamp validity and user/item referential integrity. Invalid "
            "rows are quarantined under data/errors and valid records continue. Preparation uses a global "
            "chronological 80/20 split; all derived summaries, features and models use training interactions only."
        ),
        para("3. Data quality and preparation", "Section"),
    ]

    ds = validation["datasets"]
    quality_rows = [["Dataset", "Checked", "Valid", "Quarantined"],
                    *[[name.title(), str(ds[name]["total"]), str(ds[name]["valid"]), str(ds[name]["invalid"])]
                      for name in ("users", "items", "interactions")]]
    quality_table = Table(quality_rows, colWidths=[55*mm, 35*mm, 35*mm, 35*mm])
    quality_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8), ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8C2CC")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F2F5F8")]),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.extend([quality_table, Spacer(1, 5), para(
        f"The chronological cutoff is {split['cutoff_timestamp_utc']}. Training contains "
        f"{split['train_rows']:,} rows ({split['actual_train_fraction']:.1%}); test contains "
        f"{split['test_rows']:,} rows. The actual ratio may differ slightly from 80/20 when timestamps tie."
    )])
    story.extend([
        para("4. Feature management", "Section"),
        para(
            "A reusable feature layer exports user features (activity count, average rating, preferred genre, "
            "recency and source demographics), item features (genre, interaction popularity, positive interaction "
            "count and average rating), and training interaction features (rating, positive-feedback flag, UTC "
            "weekday/hour and recency relative to the training cutoff). A feature catalog is saved in CSV and Markdown."
        ),
        PageBreak(),
        para("Model training, evaluation and orchestration", "ReportTitle"),
        para("5. Recommendation models", "Section"),
        para(
            "The baseline ranks items by training positive-rating count, then total training interactions and "
            "average rating. The personalized model is item-item collaborative filtering: cosine similarity is "
            "computed between item vectors from a binary matrix of positive training ratings. Previously seen items "
            "are removed from each user's candidate list; popularity is used as a fallback/tie-breaker."
        ),
        para("6. Evaluation results", "Section"),
    ])

    metric_columns = ["model", f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", "coverage", "diversity"]
    table_data = [["Model", f"P@{k}", f"R@{k}", f"NDCG@{k}", "Coverage", "Diversity"]]
    for _, row in comparison.iterrows():
        table_data.append([str(row["model"])] + [f"{float(row[c]):.4f}" for c in metric_columns[1:]])
    result_table = Table(table_data, colWidths=[57*mm, 23*mm, 23*mm, 24*mm, 24*mm, 24*mm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.5), ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8C2CC")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F2F5F8")]),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.extend([result_table, Spacer(1, 5), para(
        f"Ranking metrics are macro-averaged across {summary['users_evaluated']:,} users with at least one positive "
        f"held-out rating for an item available in the training catalog. K={k}; relevance is rating >= "
        f"{summary['positive_rating_threshold']}. Precision@K divides hits by K, Recall@K divides hits by the "
        "user's relevant held-out items, and NDCG@K discounts hits by rank. Coverage is the fraction of training "
        "catalog items recommended to at least one evaluated user. Diversity is mean intra-list Jaccard distance "
        "over genre sets."
    )])

    # Rating-prediction metrics are separate from Top-K ranking metrics.
    def display_metric(value):
        if pd.isna(value):
            return "N/A"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{number:.4f}" if pd.notna(number) else "N/A"

    story.extend([
        para("6.1 Rating prediction evaluation (RMSE and MAE)", "Section"),
        para(
            "The personalized item-item model also estimates explicit 1–5 star ratings for held-out "
            "user–movie interactions. RMSE (root mean squared error) penalizes larger prediction errors "
            "more strongly, while MAE (mean absolute error) reports the average absolute difference between "
            "predicted and actual ratings. Lower values indicate more accurate rating predictions. These "
            "metrics are computed on held-out test ratings, not on ratings used to build the model."
        )
    ])

    rating_rows = [["Model", "RMSE", "MAE"]]
    for _, row in comparison.iterrows():
        rating_rows.append([
            str(row.get("model", "Unknown")),
            display_metric(row.get("rmse", float("nan"))),
            display_metric(row.get("mae", float("nan")))
        ])
    rating_table = Table(rating_rows, colWidths=[90*mm, 35*mm, 35*mm])
    rating_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8), ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8C2CC")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F2F5F8")]),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.extend([rating_table, Spacer(1, 4), para(
        "N/A means that the model does not produce explicit user-specific rating predictions or that "
        "the corresponding metric was unavailable in the evaluation output."
    )])
    story.extend([
        para("7. Orchestration and reproducibility", "Section"),
        para(
            "Prefect orchestrates ingestion → validation → preparation → transformation → feature generation → "
            "model training → evaluation/reporting. Task retries are configured, and the two models are trained "
            "as separate tasks. MLflow records experiment parameters, metrics and artifacts using a local tracking "
            "store. The full workflow is launched with one command documented in README.md."
        ),
        para("8. Limitations and considerations", "Section"),
        para(
            f"{summary['cold_start_positive_test_interactions_excluded']:,} positive held-out interactions involving "
            "items absent from the training catalog are excluded from ranking evaluation, and "
            f"{summary['positive_test_interactions_for_users_without_training_history']:,} positive held-out interactions "
            "belonging to users without training history are separately counted. Neither model can recommend an "
            "unseen catalog item or personalize for a user with no history. The collaborative filter is item-item "
            "and uses genre metadata only "
            "for diversity measurement, not model scoring. New users/items therefore remain cold-start cases. "
            "The MovieLens 1M data reflects historical ratings and should not be interpreted as current viewing "
            "preferences. Results depend on the chosen chronological cutoff, positive-rating threshold and K."
        ),
        para("Conclusion", "Section"),
        para("The implementation demonstrates the complete data management lifecycle and compares a global "
             "popularity baseline with a personalized collaborative-filtering recommender. The saved validation "
             "logs, feature catalog, model artifacts, evaluation table, MLflow runs and Prefect execution history "
             "provide traceable evidence for the pipeline.")
    ])
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=17*mm, leftMargin=17*mm,
                            topMargin=14*mm, bottomMargin=14*mm, title="DML Assignment 2 Final Report")
    doc.build(story)

    md = f"""# End-to-End Data Management Pipeline for a Recommendation System\n\n## Objective and dataset\nThis project implements a reproducible MovieLens 1M recommendation pipeline covering ingestion, validation, chronological preparation, feature engineering, two recommendation models, evaluation, MLflow tracking and Prefect orchestration. MovieLens 1M provides user demographics, movie genres, ratings and timestamps. Age is preserved as an age-group code.\n\n## Data quality and preparation\n- User records: {ds['users']['valid']:,} valid / {ds['users']['total']:,} checked; {ds['users']['invalid']:,} quarantined.\n- Item records: {ds['items']['valid']:,} valid / {ds['items']['total']:,} checked; {ds['items']['invalid']:,} quarantined.\n- Interaction records: {ds['interactions']['valid']:,} valid / {ds['interactions']['total']:,} checked; {ds['interactions']['invalid']:,} quarantined.\n- Chronological cutoff: {split['cutoff_timestamp_utc']}. Training: {split['train_rows']:,}; test: {split['test_rows']:,}.\n\n## Feature management\nReusable user, item and interaction features are created from training data only. Definitions and computation rules are in `feature_catalog.csv`.\n\n## Model design\n- Popularity baseline: rank by positive training interactions, total interactions and average rating.\n- Personalized: item-item collaborative filtering using cosine similarity over binary positive-rating histories. Previously seen items are excluded from recommendations.\n\n## Ranking evaluation (K={k})
{comparison.to_markdown(index=False)}

Ranking metrics are macro-averaged for users with positive held-out interactions in the training catalog. Relevance threshold: rating >= {summary['positive_rating_threshold']}. Diversity is mean intra-list Jaccard distance over genre sets. Coverage is the fraction of training catalog items recommended at least once.

## Rating prediction evaluation (RMSE and MAE)
The personalized item-item model also estimates explicit 1–5 star ratings for held-out user–movie interactions. RMSE (root mean squared error) penalizes larger errors more strongly; MAE (mean absolute error) is the average absolute prediction error. Lower values indicate more accurate rating predictions. These metrics are evaluated on held-out test ratings.

| Model | RMSE | MAE |
|---|---:|---:|
{chr(10).join(f"| {row.get('model', 'Unknown')} | {display_metric(row.get('rmse', float('nan')))} | {display_metric(row.get('mae', float('nan')))} |" for _, row in comparison.iterrows())}

N/A means that the model does not produce explicit rating predictions or that the metric was unavailable.

## Orchestration and limitations\nPrefect orchestrates all pipeline stages; MLflow tracks parameters, metrics and artifacts. {summary['cold_start_positive_test_interactions_excluded']:,} positive held-out interactions on items absent from the training catalog are excluded from ranking evaluation; {summary['positive_test_interactions_for_users_without_training_history']:,} positive held-out interactions belonging to users without training history are also separately counted. New users/items remain cold-start cases; the item-item collaborative filter does not consume genre features for scoring.\n"""
    md_path.write_text(md, encoding="utf-8")
    print(f"Generated final report: {pdf_path}")
    return {"pdf": str(pdf_path), "markdown": str(md_path)}
