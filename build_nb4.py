"""Builds notebooks/netflix_lightfm.ipynb (LightFM WARP ranking experiment).
Run once: python build_nb4.py

NOTE: LightFM needs a C/Cython build; it installs on Colab but not on the Windows
Store Python, so this notebook is the one piece not verified locally. The API usage
follows the LightFM docs; run it on Colab."""
import json

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s}
cells = []

cells.append(md(
"# Netflix Prize — LightFM (WARP) ranking experiment\n"
"\n"
"MF and kNN optimize **rating accuracy** (RMSE) and rank as a side effect. **LightFM with\n"
"WARP loss optimizes ranking directly** — so the question is whether it beats MF on\n"
"**sampled-neg MAP@10** (MF got ~0.31). It does *not* predict a star rating, so there's no\n"
"RMSE here and it can't join the rating-blend directly (it could join a *rank* blend).\n"
"\n"
"Reuses the same parse / split / metrics as the other notebooks (same `SEED`).\n"
"\n"
"> Not verified locally (LightFM needs a Cython build absent on the dev machine) — run on Colab."))

cells.append(code(
"!pip install -q lightfm\n"
"# If the build fails on a very new runtime, try:  !pip install -q lightfm --no-build-isolation"))

cells.append(md("## 0. Setup + parse + split (shared)"))
cells.append(code(
"import os, time\n"
"import numpy as np\n"
"import pandas as pd\n"
"from scipy.sparse import coo_matrix\n"
"\n"
"DATA_DIR  = '/content/netflix'      # <-- EDIT ME\n"
"CACHE_NPZ = os.path.join(DATA_DIR, 'ratings_cache.npz')\n"
"CHUNK_ROWS = 5_000_000\n"
"SUBSAMPLE_FILES = [1, 2, 3, 4]\n"
"USER_FRAC = 1.0\n"
"SEED = 42\n"
"TEST_FRAC = 0.20\n"
"RELEVANT_THRESHOLD = 3.5\n"
"\n"
"# LightFM hyperparameters\n"
"N_COMPONENTS = 100\n"
"EPOCHS_LFM   = 20\n"
"POS_THRESHOLD = 4.0     # rating >= this counts as a positive interaction (WARP is implicit)\n"
"NUM_THREADS  = 4\n"
"rng = np.random.default_rng(SEED)\n"
"print('config loaded')"))

cells.append(code(
"# --- Colab only: mount Drive and/or unzip the dataset. Skip if running locally. ---\n"
"try:\n"
"    from google.colab import drive\n"
"    drive.mount('/content/drive')\n"
"    # Example: unzip the archive you uploaded to Drive into /content/netflix\n"
"    # !mkdir -p /content/netflix\n"
"    # !unzip -o '/content/drive/MyDrive/archive.zip' -d /content/netflix\n"
"except Exception as e:\n"
"    print('Not on Colab (or skip):', e)"))

cells.append(code(
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
"    raw_user, movie, rating, date = z['user'], z['movie'], z['rating'], z['date']\n"
"else:\n"
"    pu,pm,pr,pd_ = [],[],[],[]\n"
"    for n in SUBSAMPLE_FILES:\n"
"        u,m,r,d = parse_combined(os.path.join(DATA_DIR, f'combined_data_{n}.txt'))\n"
"        pu.append(u); pm.append(m); pr.append(r); pd_.append(d)\n"
"    raw_user=np.concatenate(pu); movie=np.concatenate(pm)\n"
"    rating=np.concatenate(pr); date=np.concatenate(pd_)\n"
"    np.savez(CACHE_NPZ, user=raw_user, movie=movie, rating=rating, date=date)\n"
"\n"
"_, user = np.unique(raw_user, return_inverse=True); user = user.astype(np.int32); del raw_user\n"
"movie = (movie - 1).astype(np.int32)\n"
"if USER_FRAC < 1.0:\n"
"    keep = np.zeros(user.max()+1, bool)\n"
"    keep[rng.choice(keep.size, int(keep.size*USER_FRAC), replace=False)] = True\n"
"    mr = keep[user]; user, movie, rating, date = user[mr], movie[mr], rating[mr], date[mr]\n"
"    _, user = np.unique(user, return_inverse=True); user = user.astype(np.int32)\n"
"n_users = int(user.max())+1; n_movies = int(movie.max())+1\n"
"print(f'{len(rating):,} ratings | {n_users:,} users | {n_movies:,} movies')\n"
"\n"
"order = np.lexsort((date, user)); su = user[order]\n"
"_, st, ct = np.unique(su, return_index=True, return_counts=True)\n"
"st = st.astype(np.int32); ct = ct.astype(np.int32)\n"
"pos = np.arange(len(order), dtype=np.int32) - np.repeat(st, ct)\n"
"nt = np.where(ct >= 5, np.ceil(ct*TEST_FRAC).astype(np.int32), 0)\n"
"is_test = np.zeros(len(order), bool); is_test[order] = pos >= np.repeat(ct-nt, ct)\n"
"del order, su, pos\n"
"trm = ~is_test\n"
"u_tr, m_tr, r_tr = user[trm], movie[trm], rating[trm]\n"
"u_te, m_te, r_te = user[is_test], movie[is_test], rating[is_test]\n"
"print(f'train {trm.sum():,} | test {is_test.sum():,}')"))

cells.append(md("## 1. MAP@10 metrics (same as the MF/kNN notebooks)"))
cells.append(code(
"def map_at_k(u_te, m_te, r_te, predict_fn, k=10, max_users=20000, seed=SEED):\n"
"    pred = predict_fn(u_te, m_te)\n"
"    o = np.argsort(u_te, kind='stable'); uu, rr, pp = u_te[o], r_te[o], pred[o]\n"
"    users_u, st = np.unique(uu, return_index=True); en = np.append(st[1:], len(uu))\n"
"    rgn = np.random.default_rng(seed); sel = np.arange(len(users_u))\n"
"    if len(sel) > max_users: sel = rgn.choice(sel, max_users, replace=False)\n"
"    aps = []\n"
"    for j in sel:\n"
"        s, e = st[j], en[j]\n"
"        if e - s < 2: continue\n"
"        rel = rr[s:e] >= RELEVANT_THRESHOLD; R = int(rel.sum())\n"
"        if R == 0: continue\n"
"        rank = np.argsort(-pp[s:e], kind='stable')[:k]\n"
"        hits = 0; ap = 0.0\n"
"        for i, ri in enumerate(rank, 1):\n"
"            if rel[ri]: hits += 1; ap += hits / i\n"
"        aps.append(ap / min(k, R))\n"
"    return float(np.mean(aps)), len(aps)\n"
"\n"
"def map_at_k_sampled(predict_fn, u_tr, m_tr, u_te, m_te, r_te,\n"
"                     k=10, n_neg=100, max_users=10000, seed=SEED):\n"
"    rgn = np.random.default_rng(seed)\n"
"    o = np.argsort(u_tr, kind='stable'); su, sm = u_tr[o], m_tr[o]\n"
"    bounds = np.searchsorted(su, np.arange(n_users + 1))\n"
"    ot = np.argsort(u_te, kind='stable'); tu, tm, trr = u_te[ot], m_te[ot], r_te[ot]\n"
"    tusers, tst = np.unique(tu, return_index=True); ten = np.append(tst[1:], len(tu))\n"
"    sel = np.arange(len(tusers))\n"
"    if len(sel) > max_users: sel = rgn.choice(sel, max_users, replace=False)\n"
"    aps = []\n"
"    for j in sel:\n"
"        uidx = int(tusers[j]); s, e = tst[j], ten[j]\n"
"        t_movies, t_r = tm[s:e], trr[s:e]\n"
"        rel = t_r >= RELEVANT_THRESHOLD; R = int(rel.sum())\n"
"        if R == 0: continue\n"
"        seen = set(sm[bounds[uidx]:bounds[uidx + 1]].tolist()) | set(t_movies.tolist())\n"
"        negs = []\n"
"        while len(negs) < n_neg:\n"
"            for c in rgn.integers(0, n_movies, n_neg):\n"
"                c = int(c)\n"
"                if c not in seen:\n"
"                    negs.append(c); seen.add(c)\n"
"                    if len(negs) >= n_neg: break\n"
"        cand = np.concatenate([t_movies, np.array(negs, dtype=np.int32)])\n"
"        labels = np.concatenate([rel, np.zeros(len(negs), bool)])\n"
"        scores = predict_fn(np.full(len(cand), uidx, np.int32), cand)\n"
"        rank = np.argsort(-scores, kind='stable')[:k]\n"
"        hits = 0; ap = 0.0\n"
"        for i, ri in enumerate(rank, 1):\n"
"            if labels[ri]: hits += 1; ap += hits / i\n"
"        aps.append(ap / min(k, R))\n"
"    return float(np.mean(aps)), len(aps)"))

cells.append(md(
"## 2. Train LightFM (WARP)\n"
"\n"
"WARP needs **implicit positives**: we treat `rating >= POS_THRESHOLD` as a positive\n"
"interaction (a strong 'like'). WARP samples negatives and pushes positives above them —\n"
"directly optimizing the top of the ranking, which is what MAP@10 rewards."))
cells.append(code(
"from lightfm import LightFM\n"
"\n"
"pos = r_tr >= POS_THRESHOLD\n"
"interactions = coo_matrix((np.ones(int(pos.sum()), np.float32),\n"
"                           (u_tr[pos], m_tr[pos])), shape=(n_users, n_movies))\n"
"print(f'{interactions.nnz:,} positive interactions (rating >= {POS_THRESHOLD})')\n"
"\n"
"lfm = LightFM(loss='warp', no_components=N_COMPONENTS, random_state=SEED)\n"
"t0 = time.time()\n"
"lfm.fit(interactions, epochs=EPOCHS_LFM, num_threads=NUM_THREADS, verbose=True)\n"
"print(f'trained in {time.time()-t0:.0f}s')"))

cells.append(md("## 3. Evaluate ranking (sampled-neg MAP@10) vs baseline"))
cells.append(code(
"def predict_lightfm(u_arr, m_arr):\n"
"    return lfm.predict(np.asarray(u_arr, np.int32), np.asarray(m_arr, np.int32),\n"
"                       num_threads=NUM_THREADS)\n"
"\n"
"# popularity baseline for a reference floor (item frequency = pure popularity ranking)\n"
"pop = np.bincount(m_tr, minlength=n_movies).astype(np.float32)\n"
"def predict_pop(u_arr, m_arr):\n"
"    return pop[np.asarray(m_arr)]\n"
"\n"
"s_ho, n_ho = map_at_k(u_te, m_te, r_te, predict_lightfm)\n"
"s_sn, n_sn = map_at_k_sampled(predict_lightfm, u_tr, m_tr, u_te, m_te, r_te)\n"
"p_sn, _    = map_at_k_sampled(predict_pop,      u_tr, m_tr, u_te, m_te, r_te)\n"
"print(f'LightFM(WARP) MAP@10  held-out={s_ho:.4f} ({n_ho})  sampled-neg={s_sn:.4f} ({n_sn})')\n"
"print(f'popularity     MAP@10  sampled-neg={p_sn:.4f}')\n"
"print(f'(MF from the main notebook: sampled-neg ~0.31 — compare here)')"))

cells.append(md(
"## 4. Notes\n"
"\n"
"- **If LightFM's sampled-neg MAP@10 > MF's ~0.31**, ranking-aware training helped — report\n"
"  it as the strongest *ranking* model even though MF wins on RMSE. That RMSE-vs-MAP split is\n"
"  exactly the rating-accuracy-vs-ranking trade-off the rubric asks you to discuss.\n"
"- **Incorporating it**: LightFM scores aren't ratings, so it can't go into the RMSE ridge\n"
"  blend. Two ways to use it: (a) serve MF for predicted ratings + LightFM for the Top-K list;\n"
"  (b) a *rank-level* blend (average the per-user item ranks from MF and LightFM).\n"
"- **Hybrid**: LightFM's real edge is item/user **features** — feed movie year/decade (and\n"
"  later your IMDB genre/language join) via `item_features` to fight cold-start. That's the\n"
"  natural bridge to the hybrid-recommender optional task.\n"
"- Tune `POS_THRESHOLD` (3.5 vs 4.0), `no_components`, `epochs`; try `loss='bpr'` for contrast."))

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open("notebooks/netflix_lightfm.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote notebooks/netflix_lightfm.ipynb with", len(cells), "cells")
