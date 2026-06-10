import numpy as np, pandas as pd, os
D = r"C:\Users\jneelamegam\netflix-prize\data"
df = pd.read_csv(os.path.join(D,"combined_data_1.txt"), header=None,
                 names=['user','rating','date'], dtype={'user':str}, nrows=8_000_000)
h = df['rating'].isna()
mv = pd.to_numeric(df['user'].where(h).str.rstrip(':'), errors='coerce').ffill().fillna(1)
rows = ~h
movie = mv[rows].to_numpy(np.int32)
user = pd.to_numeric(df.loc[rows,'user']).to_numpy(np.int64)
rating = df.loc[rows,'rating'].to_numpy(np.float32)
date = df.loc[rows,'date'].to_numpy('datetime64[D]')
titles = pd.read_csv(os.path.join(D,"movie_titles.csv"), header=None, encoding='latin-1',
                     on_bad_lines='skip', names=['movie_id','year','title'], usecols=[0,1,2])
titles['year'] = pd.to_numeric(titles['year'], errors='coerce')
tinfo = titles.set_index('movie_id')
print(f'{len(rating):,} ratings | movies {movie.min()}..{movie.max()}')

agg = pd.DataFrame({'movie_id':movie,'rating':rating}).groupby('movie_id')['rating'].agg(['count','mean'])
agg = agg.join(tinfo)
agg['title_len'] = agg['title'].str.len()
agg['n_words'] = agg['title'].str.split().str.len()

# (a) avg title length of top-5 most-rated movies per release year
top5 = agg[agg['year'].notna()].sort_values('count', ascending=False).groupby('year').head(5)
byyear = top5.groupby('year').agg(avg_title_len=('title_len','mean'),
                                  avg_words=('n_words','mean'), n=('count','size'))
print('\n(a) avg title length of top-5 movies/release-year:')
print(byyear[byyear.index>=1980].iloc[::5].round(1).head(8))

# (b) comeback: old movies whose rating volume surges late in the window, highly rated
mo = date.astype('datetime64[M]')
half = np.datetime64('2004-06')
em = pd.DataFrame({'movie_id':movie, 'late':(mo>=half)})
frac_late = em.groupby('movie_id')['late'].mean().rename('frac_late')
n_mo = em.groupby('movie_id')['late'].size().rename('n')
cb = agg.join(frac_late).join(n_mo).dropna(subset=['year'])
cb = cb[(cb['year']<1990) & (cb['n']>=200) & (cb['frac_late']>0.7) & (cb['mean']>=3.7)]
print('\n(b) comeback candidates (old, high-rated, surging late):')
print(cb.sort_values('frac_late', ascending=False)[['year','n','mean','frac_late','title']].head(8).to_string())

# (c) heavy raters (>100) vs newbies (<10): favorite movies
uc = pd.Series(user).value_counts()
heavy = set(uc[uc>100].index); newb = set(uc[uc<10].index)
um = pd.DataFrame({'user':user,'movie_id':movie,'rating':rating})
def faves(group):
    g = um[um['user'].isin(group)]
    s = g.groupby('movie_id')['rating'].agg(['count','mean'])
    return s[s['count']>=30].join(tinfo['title']).sort_values('count', ascending=False).head(8)
print(f'\n(c) heavy raters ({len(heavy):,}) favorites:'); print(faves(heavy)[['count','mean','title']].to_string())
print(f'\n(c) newbie ({len(newb):,}) favorites:');        print(faves(newb)[['count','mean','title']].to_string())

# CSV: top-25 highest-rated movies per month (>=50 ratings/month)
cm = pd.DataFrame({'movie_id':movie,'mo':mo,'rating':rating})
g = cm.groupby(['mo','movie_id'])['rating'].agg(['size','mean']).reset_index()
g = g[g['size']>=50]
g['rank'] = g.groupby('mo')['mean'].rank(method='first', ascending=False)
top = (g[g['rank']<=25]
       .merge(tinfo.reset_index()[['movie_id','year','title']], on='movie_id', how='left')
       .sort_values(['mo','rank'])
       .rename(columns={'size':'n_ratings','mean':'avg_rating'}))
print(f'\nCSV: {top["mo"].nunique()} months x up to 25 -> {len(top)} rows')
print(top[['mo','rank','title','year','n_ratings','avg_rating']].head(6).to_string(index=False))
