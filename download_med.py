import sys
import io
import requests
import pandas as pd
import zipfile
import StringIO
import os
import shutil

url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

r = requests.get(url)

z = zipfile.ZipFile(StringIO.StringIO(r.content))

df = pd.read_csv(z.open("ml-1m/ratings.dat"), sep='::', header=None)
df.columns = ["userId", "movieId", "rating", "timestamp"]

user_id_map = {user_id:i for i,user_id in enumerate(list(set(df['userId'])))}
movie_id_map = {movie_id:i for i,movie_id in enumerate(list(set(df['movieId'])))}

df['userId'] = df['userId'].apply(lambda u: user_id_map[u])
df['movieId'] = df['movieId'].apply(lambda m: movie_id_map[m])

if os.path.isdir("data"):
    shutil.rmtree("data")

os.makedirs("data")

df.to_csv("data/ratings_med.csv", index=False)
