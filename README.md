# Netflix Prize — Personalized Recommendation System

Recommendation system on the [Netflix Prize dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
(100M ratings, 480K users, 17.7K movies). Predicts unseen ratings and generates
Top-K recommendations, evaluated with **RMSE** and **MAP@10**.

## Approach

Built in layers, each a model we can compare:

1. **Bias baseline** — `r̂ = μ + b_u + b_i` (regularized). Backbone for everything else.
2. **Biased matrix factorization** — `r̂ = μ + b_u + b_i + pᵤ·qᵢ`, SGD over observed
   entries (PyTorch, GPU). The low-rank model.
3. **Item-item kNN** (`netflix_knn.ipynb`) — cosine similarity on baseline residuals with
   co-rating shrinkage + denominator shrinkage, for the mandatory comparison.
4. **Ensemble blend** — ridge over model predictions. The historical Prize lesson:
   blends beat any single model.

## Repo structure

```
netflix-prize/
├── notebooks/
│   └── netflix_prize.ipynb     # baseline + MF + evaluation + Top-K (temperature) + blend
│   └── netflix_knn.ipynb       # item-kNN comparison model (built by build_nb2.py)
│   └── netflix_eda.ipynb       # extended EDA (built by build_nb3.py)
├── build_nb2.py / build_nb3.py # regenerate the notebooks above
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

For fast local iteration, set `SUBSAMPLE_FILES=[1]` and `USER_FRAC=0.35` in the config cell.

## Evaluation

- **RMSE** — rating-prediction accuracy.
- **MAP@10** — ranking quality; an item is relevant if its actual rating ≥ 3.5.
  Reported with **sampled negatives** (rank held-out items against 100 random unseen
  movies, NCF-style) — the honest catalog-style metric. A held-out-only variant is
  also printed for contrast (it is inflated, since most held-out items clear the 3.5 bar).

**Split:** per-user temporal holdout — each user's most recent 20% of ratings are the
test set (predict the future from the past).

### Results

| Model | RMSE | MAP@10 (sampled-neg) |
| --- | --- | --- |
| Matrix factorization (MF) | **0.86** | 0.31 |
| Item-item kNN | 0.88 | **0.34** |

**Rating accuracy ≠ ranking quality.** MF wins on RMSE (0.86 vs 0.88), but item-kNN is the
**better ranking model** — it gets a higher MAP@10 (0.34 vs 0.31). MF predicts the rating
value more precisely, while the neighborhood model orders each user's top items better. This
is exactly the rating-vs-ranking trade-off the brief asks us to discuss: the model you pick
depends on whether the product needs accurate scores or a good Top-K list.

## Key findings (EDA)

- **Movie title length shows no consistent trend over release years** — title length is not
  a useful signal for popularity or era.
- **Cinephiles vs. newbies have distinct taste.** Heavy raters (>100 ratings) skew toward
  **franchise entries and classic films**; newbies (<10 ratings) skew toward **rom-coms** and
  recent mainstream hits. (Motivates cold-start handling and explains why personalization
  helps power users most.)
- **Star Wars films are the highest-rated 1980s movies** as rated in the 2000s — a clear case
  of older titles sustaining strong appeal decades after release.
