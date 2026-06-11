"""Builds notebooks/netflix_eda.ipynb (extended EDA + top-25/month CSV export).
Run once: python build_nb3.py"""
import json

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s}
cells = []

cells.append(md(
"# Netflix Prize — Extended EDA\n"
"\n"
"Standalone analyses (no model needed) that go beyond the basic EDA in the MF notebook:\n"
"1. Average **title length** of the top-5 movies per release year\n"
"2. **Comeback** movies — old titles whose ratings surge late in the window\n"
"3. Tastes of **heavy raters (>100)** vs **newbies (<10)**\n"
"4. CSV export: **top-25 highest-rated movies per month** (for an IMDB join)\n"
"\n"
"Note: rating timestamps span ~2000-2005; movie *release years* go back decades — a\n"
"separate field. All analyses are vectorized with `np.bincount` so they run on the full\n"
"100M ratings. Reuses the same `ratings_cache.npz` as the model notebooks."))

cells.append(md("## 0. Setup + load"))
cells.append(code(
"import os, time\n"
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"\n"
"DATA_DIR  = '/content/netflix'      # <-- EDIT ME\n"
"CACHE_NPZ = os.path.join(DATA_DIR, 'ratings_cache.npz')\n"
"CHUNK_ROWS = 5_000_000\n"
"SUBSAMPLE_FILES = [1, 2, 3, 4]\n"
"\n"
"def parse_combined(path, chunksize=CHUNK_ROWS):\n"
"    us, ms, rs, ds = [], [], [], []; cur = 0\n"
"    for chunk in pd.read_csv(path, header=None, names=['user','rating','date'],\n"
"                             dtype={'user': str}, chunksize=chunksize):\n"
"        is_hdr = chunk['rating'].isna()\n"
"        mv = pd.to_numeric(chunk['user'].where(is_hdr).str.rstrip(':'),\n"
"                           errors='coerce').ffill().fillna(cur)\n"
"        r = ~is_hdr\n"
"        us.append(pd.to_numeric(chunk.loc[r,'user']).to_numpy(np.int32))\n"
"        ms.append(mv[r].to_numpy(np.int16)); rs.append(chunk.loc[r,'rating'].to_numpy(np.int8))\n"
"        ds.append(chunk.loc[r,'date'].to_numpy('datetime64[D]')); cur = int(mv.iloc[-1])\n"
"    return (np.concatenate(us), np.concatenate(ms), np.concatenate(rs), np.concatenate(ds))\n"
"\n"
"if os.path.exists(CACHE_NPZ):\n"
"    z = np.load(CACHE_NPZ)\n"
"    raw_user, movie1, rating, date = z['user'], z['movie'], z['rating'], z['date']\n"
"else:\n"
"    pu,pm,pr,pd_ = [],[],[],[]\n"
"    for n in SUBSAMPLE_FILES:\n"
"        u,m,r,d = parse_combined(os.path.join(DATA_DIR, f'combined_data_{n}.txt'))\n"
"        pu.append(u); pm.append(m); pr.append(r); pd_.append(d)\n"
"    raw_user=np.concatenate(pu); movie1=np.concatenate(pm)\n"
"    rating=np.concatenate(pr); date=np.concatenate(pd_)\n"
"    np.savez(CACHE_NPZ, user=raw_user, movie=movie1, rating=rating, date=date)\n"
"\n"
"movie = (movie1.astype(np.int32) - 1)            # 0-based\n"
"n_movies = int(movie.max()) + 1\n"
"_, user = np.unique(raw_user, return_inverse=True); user = user.astype(np.int32)\n"
"n_users = int(user.max()) + 1\n"
"rating_f = rating.astype(np.float32)\n"
"print(f'{len(rating):,} ratings | {n_users:,} users | {n_movies:,} movies')\n"
"print('rating dates:', date.min(), '->', date.max())\n"
"\n"
"titles = pd.read_csv(os.path.join(DATA_DIR,'movie_titles.csv'), header=None,\n"
"                     encoding='latin-1', on_bad_lines='skip',\n"
"                     names=['movie_id','year','title'], usecols=[0,1,2])\n"
"titles['year'] = pd.to_numeric(titles['year'], errors='coerce')\n"
"# per-movie tables aligned to 0-based index\n"
"mcount = np.bincount(movie, minlength=n_movies)\n"
"msum   = np.bincount(movie, weights=rating_f, minlength=n_movies)\n"
"mmean  = np.divide(msum, mcount, out=np.zeros_like(msum), where=mcount>0)\n"
"info = titles.set_index('movie_id')\n"
"year_of   = info['year'].reindex(np.arange(1, n_movies+1)).to_numpy()\n"
"title_arr = info['title'].reindex(np.arange(1, n_movies+1)).to_numpy()"))

cells.append(md(
"## 1. Average title length of the top-5 movies per release year\n"
"\n"
"For each release year, take the 5 most-rated movies and average their title length."))
cells.append(code(
"tl = pd.DataFrame({'year': year_of, 'count': mcount, 'title': title_arr})\n"
"tl['title_len'] = tl['title'].str.len()\n"
"tl['n_words']   = tl['title'].str.split().str.len()\n"
"tl = tl[tl['year'].notna() & tl['title'].notna()]\n"
"top5 = tl.sort_values('count', ascending=False).groupby('year').head(5)\n"
"byyear = top5.groupby('year').agg(title_len=('title_len','mean'),\n"
"                                  words=('n_words','mean'), n=('count','size'))\n"
"byyear = byyear[byyear['n'] >= 5]                 # years with a full top-5\n"
"\n"
"fig, ax = plt.subplots(1, 2, figsize=(13, 3.4))\n"
"ax[0].plot(byyear.index, byyear['title_len'], marker='.')\n"
"ax[0].set_title('Avg title length (chars) of top-5 movies / release year'); ax[0].set_xlabel('year')\n"
"ax[1].plot(byyear.index, byyear['words'], marker='.', color='C1')\n"
"ax[1].set_title('Avg title length (words)'); ax[1].set_xlabel('year')\n"
"plt.tight_layout(); plt.show()\n"
"print(byyear.round(2).tail(12))"))

cells.append(md(
"## 2. Comeback movies — old titles that surged late in the window\n"
"\n"
"Netflix ratings span ~2000-2005, so an *old* film rated heavily here is being watched\n"
"20+ years after release. A **comeback** = old release year + a real rating base early in\n"
"the window + a strong surge in the second half + high average rating. `surge = late/early`.\n"
"\n"
"> We can't see original box-office here (no external data), so 'failure -> comeback' is a\n"
"> proxy. Joining IMDB/box-office data (see Section 4) would let you confirm the flop part."))
cells.append(code(
"mo = date.astype('datetime64[M]')\n"
"half = mo.min() + (mo.max() - mo.min()) // 2\n"
"early = mo < half; late = ~early\n"
"ec = np.bincount(movie[early], minlength=n_movies)\n"
"lc = np.bincount(movie[late],  minlength=n_movies)\n"
"surge = lc / np.maximum(ec, 1)\n"
"cb = pd.DataFrame({'movie_id': np.arange(1, n_movies+1), 'year': year_of,\n"
"                   'title': title_arr, 'n': mcount, 'avg': mmean.round(3),\n"
"                   'early': ec, 'late': lc, 'surge': surge.round(2)})\n"
"cb = cb[(cb['year'] < 1985) & (cb['early'] >= 20) & (cb['surge'] >= 3) & (cb['avg'] >= 3.7)]\n"
"print('Top comeback candidates (old, well-rated, accelerating):')\n"
"print(cb.sort_values('surge', ascending=False)\n"
"        [['year','title','n','avg','early','late','surge']].head(20).to_string(index=False))"))

cells.append(md(
"## 3. Tastes of heavy raters (>100 ratings) vs newbies (<10)\n"
"\n"
"Popularity is normalized **within** each group (share of the group's users who rated the\n"
"movie), so the comparison isn't dominated by group size. Surfaces what cinephiles vs\n"
"casual users actually watch."))
cells.append(code(
"ucounts = np.bincount(user, minlength=n_users)\n"
"row_heavy = ucounts[user] > 100\n"
"row_newb  = ucounts[user] < 10\n"
"n_heavy = int((ucounts > 100).sum()); n_newb = int((ucounts < 10).sum())\n"
"\n"
"def group_top(mask, n_group, k=12, min_ratings=50):\n"
"    c = np.bincount(movie[mask], minlength=n_movies)\n"
"    s = np.bincount(movie[mask], weights=rating_f[mask], minlength=n_movies)\n"
"    m = np.divide(s, c, out=np.zeros_like(s), where=c>0)\n"
"    keep = np.where(c >= min_ratings)[0]\n"
"    top = keep[np.argsort(-(c[keep] / n_group))][:k]\n"
"    return pd.DataFrame({'title': title_arr[top], 'year': year_of[top],\n"
"                         'raters': c[top], 'reach_%': (100*c[top]/n_group).round(2),\n"
"                         'avg': m[top].round(2)})\n"
"print(f'HEAVY raters ({n_heavy:,}) — favorites:')\n"
"print(group_top(row_heavy, n_heavy).to_string(index=False))\n"
"print(f'\\nNEWBIES ({n_newb:,}) — favorites:')\n"
"print(group_top(row_newb, n_newb).to_string(index=False))"))

cells.append(md(
"## 4. Export: top-25 highest-rated movies per month -> CSV\n"
"\n"
"For each month, the 25 movies with the highest average rating among those with at least\n"
"`MIN_RATINGS` ratings that month. Saved to `DATA_DIR/top25_per_month.csv` — join it to\n"
"IMDB on `(title, year)` for language / genre / theme analysis.\n"
"\n"
"On Colab, download with: `from google.colab import files; files.download(out_path)`."))
cells.append(code(
"MIN_RATINGS = 50\n"
"month_codes, month_idx = np.unique(mo, return_inverse=True)\n"
"M = len(month_codes)\n"
"key  = month_idx.astype(np.int64) * n_movies + movie\n"
"cnt  = np.bincount(key, minlength=M*n_movies).reshape(M, n_movies)\n"
"ssum = np.bincount(key, weights=rating_f, minlength=M*n_movies).reshape(M, n_movies)\n"
"mean = np.divide(ssum, cnt, out=np.zeros_like(ssum), where=cnt>0)\n"
"\n"
"rows = []\n"
"for mi in range(M):\n"
"    valid = np.where(cnt[mi] >= MIN_RATINGS)[0]\n"
"    if len(valid) == 0:\n"
"        continue\n"
"    order = valid[np.argsort(-mean[mi, valid])][:25]   # ties broken by index\n"
"    for rank, mid in enumerate(order, 1):\n"
"        rows.append((str(month_codes[mi]), rank, mid + 1,\n"
"                     round(float(mean[mi, mid]), 4), int(cnt[mi, mid])))\n"
"out = pd.DataFrame(rows, columns=['month','rank','movie_id','avg_rating','n_ratings'])\n"
"out = out.merge(titles[['movie_id','year','title']], on='movie_id', how='left')\n"
"out = out[['month','rank','movie_id','title','year','n_ratings','avg_rating']]\n"
"out_path = os.path.join(DATA_DIR, 'top25_per_month.csv')\n"
"out.to_csv(out_path, index=False)\n"
"print(f'wrote {len(out):,} rows ({M} months) -> {out_path}')\n"
"out.head(10)"))

cells.append(md(
"## Notes for the report\n"
"\n"
"- **Title length**: feeds EDA (15%) and 'interesting observations'. Pair with a\n"
"  popularity trend to argue whether snappy titles correlate with reach.\n"
"- **Comebacks**: the surge metric + the IMDB join (box office vs. Netflix-era ratings)\n"
"  is a strong 'innovation' angle.\n"
"- **Heavy vs newbie**: motivates cold-start (newbies cluster on a few recent hits) and\n"
"  explains why personalization helps power users more.\n"
"- **top25_per_month.csv**: the IMDB join unlocks language / genre / theme breakdowns the\n"
"  raw dataset can't give."))

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open("notebooks/netflix_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote notebooks/netflix_eda.ipynb with", len(cells), "cells")
