import json
from ast import literal_eval
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

# Load data (built by build_movies.py; older CSVs without the newer columns still work)
app_dir = Path(__file__).parent
df = pd.read_csv(app_dir / "movies.csv")

for col, default in [("poster_path", ""), ("original_language", "en"), ("providers", "{}"), ("director", "")]:
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default)

df["title"] = df["title"].astype(str)
df["overview"] = df["overview"].fillna("")
df["runtime"] = df["runtime"].fillna(0).astype(int)
df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
df["genre_list"] = df["genres"].apply(lambda s: literal_eval(s) if isinstance(s, str) else [])
df["cast_list"] = df["cast"].apply(lambda s: literal_eval(s) if isinstance(s, str) else [])
df["provider_map"] = df["providers"].apply(lambda s: json.loads(s) if isinstance(s, str) and s else {})


def pretty(s):
    """Old CSVs were all-lowercase; newer ones already have proper capitalisation."""
    return s if any(ch.isupper() for ch in s) else s.title()


df["title_pretty"] = df["title"].apply(pretty)

# IMDB-style weighted rating: pulls movies with few votes towards the mean
_C = df["vote_average"].mean()
_m = df["vote_count"].quantile(0.60)
df["wr"] = (df["vote_count"] / (df["vote_count"] + _m)) * df["vote_average"] + (
    _m / (df["vote_count"] + _m)
) * _C
df["wr_norm"] = (df["wr"] - df["wr"].min()) / (df["wr"].max() - df["wr"].min())

genres = ['action', 'adventure', 'animation', 'comedy', 'crime', 'documentary', 'drama', 'family', 'fantasy',
          'history', 'horror', 'music', 'mystery', 'romance', 'science fiction', 'thriller', 'war', 'western']
moods = ['', 'Afraid', 'Angry', 'Anxious', 'Brave', 'Cheerful', 'Creative', 'Happy', 'Lonely', 'Sad', 'Relaxed', 'Energetic', 'Tired', 'Curious']
mood_dict = {'Afraid': 'comedy', 'Angry': 'adventure', 'Anxious': 'comedy', 'Brave': 'horror', 'Cheerful': 'action',
             'Creative': 'fantasy', 'Happy': 'adventure', 'Lonely': 'comedy', 'Sad': 'comedy', 'Relaxed': 'comedy',
             'Energetic': 'action', 'Tired': 'family', 'Curious': 'mystery'}

countries = {"IT": "Italy", "US": "United States", "GB": "United Kingdom", "DE": "Germany", "FR": "France", "ES": "Spain"}

_lang_names = {"en": "English", "it": "Italian", "fr": "French", "es": "Spanish", "de": "German", "ja": "Japanese",
               "ko": "Korean", "zh": "Chinese", "hi": "Hindi", "ru": "Russian", "sv": "Swedish", "pt": "Portuguese",
               "cn": "Cantonese", "pl": "Polish", "da": "Danish", "tr": "Turkish", "th": "Thai", "nl": "Dutch"}
languages = {"": "Any language"}
for _code in df["original_language"].value_counts().index[:12]:
    languages[_code] = _lang_names.get(_code, _code.upper())

year_min = int(df.loc[df["year"] > 0, "year"].min())
year_max = int(df["year"].max())
runtime_max = int(min(240, df["runtime"].quantile(0.99)))

# Movie choices: value = row position, label = "Title (year)" so remakes can be told apart
movie_choices = {"": ""}
for _i, _r in df.sort_values("title_pretty").iterrows():
    movie_choices[str(_i)] = f"{_r['title_pretty']} ({_r['year']})" if _r["year"] else _r["title_pretty"]

# Overview (TF-IDF) and metadata "soup" (keywords, cast, director, genres) vectors.
# Similarity is computed per query from a single row, so no n x n matrix is kept in memory.
tfidf_matrix = TfidfVectorizer(stop_words="english").fit_transform(df["overview"])  # rows are L2-normalised
count_matrix = normalize(CountVectorizer(stop_words="english").fit_transform(df["soup"].fillna("")))


def similarity(idx):
    """Blend of overview and soup cosine similarity between movie `idx` and every movie."""
    s1 = (tfidf_matrix @ tfidf_matrix[idx].T).toarray().ravel()
    s2 = (count_matrix @ count_matrix[idx].T).toarray().ravel()
    return 0.5 * s1 + 0.5 * s2
