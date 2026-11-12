# End-to-End Data Management Pipeline for a Recommendation System

## Objective and dataset
This project implements a reproducible MovieLens 1M recommendation pipeline covering ingestion, validation, chronological preparation, feature engineering, two recommendation models, evaluation, MLflow tracking and Prefect orchestration. MovieLens 1M provides user demographics, movie genres, ratings and timestamps. Age is preserved as an age-group code.

## Data quality and preparation
- User records: 6,040 valid / 6,040 checked; 0 quarantined.
- Item records: 3,883 valid / 3,883 checked; 0 quarantined.
- Interaction records: 1,000,207 valid / 1,000,214 checked; 7 quarantined.
- Chronological cutoff: 2000-12-02T14:52:18+00:00. Training: 800,168; test: 200,039.

## Feature management
Reusable user, item and interaction features are created from training data only. Definitions and computation rules are in `feature_catalog.csv`.

## Model design
- Popularity baseline: rank by positive training interactions, total interactions and average rating.
- Personalized: item-item collaborative filtering using cosine similarity over binary positive-rating histories. Previously seen items are excluded from recommendations.

## Ranking evaluation (K=5)
| model                             |   precision_at_5 |   recall_at_5 |   ndcg_at_5 |   coverage |   diversity |   users_evaluated |   unique_items_recommended |   catalog_size |      rmse |       mae |   rating_predictions_evaluated |
|:----------------------------------|-----------------:|--------------:|------------:|-----------:|------------:|------------------:|---------------------------:|---------------:|----------:|----------:|-------------------------------:|
| Popularity baseline               |         0.231551 |     0.0313954 |    0.241696 |  0.0245767 |    0.746622 |              1122 |                         90 |           3662 | nan       | nan       |                              0 |
| Item-item collaborative filtering |         0.248307 |     0.0384465 |    0.262408 |  0.123703  |    0.72495  |              1122 |                        453 |           3662 |   1.07136 |   0.82065 |                         104448 |

Ranking metrics are macro-averaged for users with positive held-out interactions in the training catalog. Relevance threshold: rating >= 4.0. Diversity is mean intra-list Jaccard distance over genre sets. Coverage is the fraction of training catalog items recommended at least once.

## Rating prediction evaluation (RMSE and MAE)
The personalized item-item model also estimates explicit 1–5 star ratings for held-out user–movie interactions. RMSE (root mean squared error) penalizes larger errors more strongly; MAE (mean absolute error) is the average absolute prediction error. Lower values indicate more accurate rating predictions. These metrics are evaluated on held-out test ratings.

| Model | RMSE | MAE |
|---|---:|---:|
| Popularity baseline | N/A | N/A |
| Item-item collaborative filtering | 1.0714 | 0.8207 |

N/A means that the model does not produce explicit rating predictions or that the metric was unavailable.

## Orchestration and limitations
Prefect orchestrates all pipeline stages; MLflow tracks parameters, metrics and artifacts. 55 positive held-out interactions on items absent from the training catalog are excluded from ranking evaluation; 55,969 positive held-out interactions belonging to users without training history are also separately counted. New users/items remain cold-start cases; the item-item collaborative filter does not consume genre features for scoring.
