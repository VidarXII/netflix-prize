"""Torch-free barebones sanity check: parse -> EDA -> split -> baseline -> MAP@10.
Mirrors the notebook's non-MF logic so we can validate it on real local data.
"""
import os, time
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FRAC = 1.0                 # fraction of file-1 rows to keep
TEST_FRAC = 0.20
RELEVANT_THRESHOLD = 3.5
SEED = 42
rng = np.random.default_rng(SEED)


def parse_combined(path):
    df = pd.read_csv(path, header=None, names=['user', 'rating', 'date'],
                     dtype={'user': str})
    is_movie = df['rating'].isna()
    df['movie'] = df['user'].where(is_movie).str.rstrip(':').ffill()
    df = df[~is_movie].copy()
    return (df['user'].to_numpy(dtype=np.int64),
            df['movie'].to_numpy(dtype=np.int32),
            df['rating'].to_numpy(dtype=np.float32),
            df['date'].to_numpy(dtype='datetime64[D]'))


t = time.time()
raw_user, movie, rating, date = parse_combined(os.path.join(DATA_DIR, 'combined_data_1.txt'))
print(f'parsed {len(rating):,} ratings in {time.time()-t:.0f}s')

if FRAC < 1.0:
    keep = rng.random(len(rating)) < FRAC
    raw_user, movie, rating, date = raw_user[keep], movie[keep], rating[keep], date[keep]

uniq_users, user = np.unique(raw_user, return_inverse=True)
user = user.astype(np.int32)
movie = (movie - 1).astype(np.int32)
n_users, n_movies = len(uniq_users), int(movie.max()) + 1
sparsity = len(rating) / (n_users * n_movies)
print(f'{n_users:,} users | {n_movies:,} movies | mean {rating.mean():.4f} | density {sparsity:.4%}')

# per-user temporal split
order = np.lexsort((date, user))
su = user[order]
_, starts, counts = np.unique(su, return_index=True, return_counts=True)
pos = np.arange(len(order)) - np.repeat(starts, counts)
n_test = np.where(counts >= 5, np.ceil(counts * TEST_FRAC).astype(int), 0)
thresh = np.repeat(counts - n_test, counts)
is_test = np.zeros(len(order), bool)
is_test[order] = pos >= thresh
tr = ~is_test
u_tr, m_tr, r_tr = user[tr], movie[tr], rating[tr]
u_te, m_te, r_te = user[is_test], movie[is_test], rating[is_test]
print(f'train {tr.sum():,} | test {is_test.sum():,}')


def fit_biases(u, m, r, n_u, n_m, reg=10.0, n_iter=10):
    mu = r.mean()
    bu = np.zeros(n_u, np.float32); bi = np.zeros(n_m, np.float32)
    uc = np.bincount(u, minlength=n_u); ic = np.bincount(m, minlength=n_m)
    for _ in range(n_iter):
        bi = np.bincount(m, weights=r - mu - bu[u], minlength=n_m) / (ic + reg)
        bu = np.bincount(u, weights=r - mu - bi[m], minlength=n_u) / (uc + reg)
    return mu, bu, bi


mu, bu, bi = fit_biases(u_tr, m_tr, r_tr, n_users, n_movies)
pred = np.clip(mu + bu[u_te] + bi[m_te], 1, 5)
print(f'baseline test RMSE: {np.sqrt(np.mean((pred - r_te)**2)):.4f}')


def map_at_k(u_te, m_te, r_te, pred, k=10, max_users=20000):
    o = np.argsort(u_te, kind='stable')
    uu, rr, pp = u_te[o], r_te[o], pred[o]
    users_u, st = np.unique(uu, return_index=True)
    en = np.append(st[1:], len(uu))
    sel = np.arange(len(users_u))
    if len(sel) > max_users:
        sel = rng.choice(sel, max_users, replace=False)
    aps = []
    for j in sel:
        s, e = st[j], en[j]
        if e - s < 2:
            continue
        rel = rr[s:e] >= RELEVANT_THRESHOLD
        R = int(rel.sum())
        if R == 0:
            continue
        rank = np.argsort(-pp[s:e], kind='stable')[:k]
        hits = 0; ap = 0.0
        for i, ri in enumerate(rank, 1):
            if rel[ri]:
                hits += 1; ap += hits / i
        aps.append(ap / min(k, R))
    return float(np.mean(aps)), len(aps)


score, nu = map_at_k(u_te, m_te, r_te, pred)
print(f'baseline MAP@10 (held-out only):   {score:.4f} (over {nu} users)')


def map_at_k_sampled(predict_fn, u_tr, m_tr, u_te, m_te, r_te, n_users, n_movies,
                     k=10, n_neg=100, max_users=5000):
    """Honest Top-K eval: rank each user's held-out items against n_neg random
    unseen movies. Relevant = held-out item with rating >= RELEVANT_THRESHOLD;
    sampled negatives are non-relevant."""
    o = np.argsort(u_tr, kind='stable')                 # train movies by user (CSR-style)
    su, sm = u_tr[o], m_tr[o]
    bounds = np.searchsorted(su, np.arange(n_users + 1))
    ot = np.argsort(u_te, kind='stable')                # test grouped by user
    tu, tm, trr = u_te[ot], m_te[ot], r_te[ot]
    tusers, tst = np.unique(tu, return_index=True)
    ten = np.append(tst[1:], len(tu))
    sel = np.arange(len(tusers))
    if len(sel) > max_users:
        sel = rng.choice(sel, max_users, replace=False)
    aps = []
    for j in sel:
        uidx = int(tusers[j]); s, e = tst[j], ten[j]
        t_movies, t_r = tm[s:e], trr[s:e]
        rel = t_r >= RELEVANT_THRESHOLD
        R = int(rel.sum())
        if R == 0:
            continue
        seen = set(sm[bounds[uidx]:bounds[uidx + 1]].tolist()) | set(t_movies.tolist())
        negs = []
        while len(negs) < n_neg:                        # rejection-sample unseen movies
            for c in rng.integers(0, n_movies, n_neg):
                c = int(c)
                if c not in seen:
                    negs.append(c); seen.add(c)
                    if len(negs) >= n_neg:
                        break
        cand = np.concatenate([t_movies, np.array(negs, dtype=np.int32)])
        labels = np.concatenate([rel, np.zeros(len(negs), bool)])
        scores = predict_fn(np.full(len(cand), uidx, np.int32), cand)
        rank = np.argsort(-scores, kind='stable')[:k]
        hits = 0; ap = 0.0
        for i, ri in enumerate(rank, 1):
            if labels[ri]:
                hits += 1; ap += hits / i
        aps.append(ap / min(k, R))
    return float(np.mean(aps)), len(aps)


def predict_baseline(u, m):
    return np.clip(mu + bu[u] + bi[m], 1, 5)


score_s, nu_s = map_at_k_sampled(predict_baseline, u_tr, m_tr, u_te, m_te, r_te,
                                 n_users, n_movies)
print(f'baseline MAP@10 (100 negatives):  {score_s:.4f} (over {nu_s} users)')
print('\nSANITY CHECK PASSED')
