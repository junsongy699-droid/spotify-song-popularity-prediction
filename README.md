# Spotify Song Popularity Prediction

F20DL · Data Mining and Machine Learning · 2026/2027 · Group Coursework

Predict how popular a Spotify song becomes, using its **audio features and metadata**. This repository is the group's coursework code base and tracks requirements **R1–R4** and deliverables **D1–D4** of the module brief.

---

## 1. Research question (R1)

> Can we predict a song's popularity from its audio features and metadata?

- **O1 — Regression**: predict `track_popularity` (0–100).
- **O2 — Classification**: detect `explicit` vs non-explicit tracks (222 : 778, imbalanced).
- **O3 — Unsupervised**: k-means clustering of audio features (elbow method + silhouette score).
- **O4 — Comparison**: every model evaluated under one framework — same split, same metrics, same seed.

## 2. Dataset (D1)

- File: `data/spotifydataset.csv` (selected among three candidates — richest features, dual tasks, right size for a 3-member team).
- 1,000 tracks, 1971–2024; 23 attributes, 11 audio features; target `track_popularity` (0–100).
- Known issues handled in preprocessing (R2): 163 missing `genres`; imbalanced `explicit`.
- EDA finding: `artist_popularity` is perfectly correlated with `track_popularity` (r = 1.0, i.e. a target-leakage column) and is therefore **excluded from modelling** (see `src/load_data.py`).

## 3. Repository layout

```
.
├── data/            spotifydataset.csv
├── src/             package: load_data, eda, clustering, baseline_models, mlp_model, evaluate
├── scripts/         run_pipeline.py (end-to-end reproduction)
├── report/          D2 report placeholder (final report ≤ 6 pages, due 25 Nov 2026, 15:30)
├── results/         generated figures and metrics
├── requirements.txt
└── README.md
```

## 4. Setup & reproduction

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py
```

The pipeline runs: preprocessing → EDA (figures to `results/`) → k-means clustering → ≥3 baselines (R3) → MLP (R4) → prints a metric summary.

## 5. Method (R2–R4)

| Requirement | What the code does |
|---|---|
| R2 | Preprocessing (missing genres, feature encoding), exploratory analysis & visualisation, k-means clustering (elbow + silhouette) |
| R3 | ≥3 baselines covering the course families: Decision Tree, Naive Bayes, Linear models, Perceptron, kNN, Ensembles — same split & seed |
| R4 | MLP (multi-layer perceptron) regression & classification via scikit-learn |

## 6. Evaluation metrics

- Regression: RMSE · MAE · R²
- Classification: Accuracy · Precision · Recall · F1
- Clustering: silhouette score

## 7. Deliverables timeline

| Milestone | Due |
|---|---|
| D1 Project Pitch | Week 4 (this deck) |
| D2 Report (≤ 6 pages PDF) + Code | **25 Nov 2026, 15:30 (Week 11)** |
| D3 Mini-viva | Week 12 |
| D4 Peer assessment | Week 12 |

## 8. Team

Group ___ · 3 members

- Member 1 · Lead + Data & EDA (preprocessing, EDA, clustering)
- Member 2 · Modelling (R3 baselines + R4 MLP, evaluation)
- Member 3 · Report & Repo (6-page report, repository, README, viva materials)

*Accessibility: the repository is private during development; before the D2 deadline it will be made visible to the teaching staff (shared/collaborators or public), per the coursework brief.*
