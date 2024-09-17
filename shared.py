from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import pandas as pd

# Load data
app_dir = Path(__file__).parent
df = pd.read_csv(app_dir / "movies.csv")
niche_df = df[df['niche']==1]
not_niche_df = df[df['niche']==0]

genres = ['action', 'adventure', 'animation', 'comedy', 'crime', 'drama', 'family', 'fantasy', 'history', 'horror', 'mystery', 'romance', 'thriller', 'war', 'western']
moods = ['','Afraid', 'Angry', 'Anxious', 'Brave', 'Cheerful', 'Creative', 'Happy', 'Lonely', 'Sad', 'Relaxed', 'Energetic', 'Tired', 'Curious']
mood_dict={'':'', 'Normal':'drama', 'Afraid':'comedy', 'Angry':'adventure', 'Anxious':'comedy', 'Brave':'horror', 'Cheerful':'action',
            'Creative':'fantasy', 'Happy':'adventure', 'Lonely':'comedy', 'Sad':'comedy', 'Relaxed':'comedy', 'Energetic':'action', 'Tired':'family', 'Curious':'mystery'}
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

movie_list = df["title"]
movie_list.loc[-1] = ''  # adding a row
movie_list.index = movie_list.index + 1  # shifting index
movie_list = movie_list.sort_index()  # sorting by index
movie_list=list(movie_list.values)

#Remove all stop words
tfidf = TfidfVectorizer(stop_words='english')

df['overview'] = df['overview'].fillna('')

#fit and transform data
tfidf_matrix = tfidf.fit_transform(df['overview'])
# Compute cosine similarity
cosine_sim1 = linear_kernel(tfidf_matrix, tfidf_matrix)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)