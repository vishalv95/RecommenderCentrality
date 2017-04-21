import sys
import pandas as pd

urls = {"ml-small": "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"}

df = pd.read_csv(urls[sys.argv[1]], compression='zip')
user_id_map = {user_id:i for i,user_id in enumerate(list(set(df['userId'])))}
movie_id_map = {movie_id:i for i,movie_id in enumerate(list(set(df['movieId'])))}

df['userId'] = df['userId'].apply(lambda u: user_id_map[u])
df['movieId'] = df['movieId'].apply(lambda m: movie_id_map[m])

df.to_csv("ratings.csv", index=False)
