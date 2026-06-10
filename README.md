# Netflix Prize — Personalized Recommendation System

Recommendation system on the [Netflix Prize dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
(100M ratings, 480K users, 17.7K movies). Predicts unseen ratings and generates
Top-K recommendations, evaluated with **RMSE** and **MAP@10**.

## Approach

Built in layers, each a model we can compare:

1. **Bias baseline** — `r̂ = μ + b_u + b_i` (regularized). Backbone for everything else.
2. **Biased matrix factorization** — `r̂ = μ + b_u + b_i + pᵤ·qᵢ`, SGD over observed
   entries (PyTorch, GPU). The low-rank model.
3. **Second model** (in a separate notebook) — item-kNN / ALS, for the mandatory comparison.
4. **Ensemble blend** — ridge over model predictions. The historical Prize lesson:
   blends beat any single model.

## Repo structure

```
netflix-prize/
├── notebooks/
│   └── netflix_prize.ipynb     # baseline + MF + evaluation + Top-K + blend
│   └── (planned) model2.ipynb  # item-kNN / ALS comparison
├── sanity_check.py             # torch-free pipeline check (parse → baseline → MAP@10)
├── data/                       # gitignored — see "Data" below
├── requirements.txt
└── README.md
```

## Data

Not committed (multi-GB). Download from Kaggle and place the files in `data/`:

```
data/
├── combined_data_1.txt   (and _2, _3, _4)
├── movie_titles.csv
├── probe.txt
└── qualifying.txt
```

## Reproduce

**Local sanity check** (no GPU, no torch needed — uses `combined_data_1.txt`):

```bash
pip install numpy pandas scikit-learn
python sanity_check.py
```

**Full pipeline** (Colab recommended for the GPU MF run):

1. Open `notebooks/netflix_prize.ipynb` in Google Colab (*File → Open notebook → GitHub*).
2. Runtime → Change runtime type → **GPU** (High-RAM for the full 100M-row parse).
3. Set `DATA_DIR` in the config cell; mount Drive / unzip the dataset.
4. Run all. Parsing caches to `ratings_cache.npz` so reruns skip it.

For fast local iteration, set `SUBSAMPLE_FILES=[1]` and `FRAC=0.2` in the config cell.

## Evaluation

- **RMSE** — rating-prediction accuracy.
- **MAP@10** — ranking quality; an item is relevant if its actual rating ≥ 3.5.
  Reported with **sampled negatives** (rank held-out items against 100 random unseen
  movies, NCF-style) — the honest catalog-style metric. A held-out-only variant is
  also printed for contrast (it is inflated, since most held-out items clear the 3.5 bar).

**Split:** per-user temporal holdout — each user's most recent 20% of ratings are the
test set (predict the future from the past).

### Current results (`combined_data_1.txt`, bias baseline)

| Metric | Value |
| --- | --- |
| RMSE | 0.930 |
| MAP@10 (sampled negatives) | 0.176 |
| MAP@10 (held-out only) | 0.793 |

MF and the second model fill in the rest of the comparison table.
