import sys
import io
import requests
import pandas as pd
import zipfile
import StringIO
import os
import shutil
import numpy as np

url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

r = requests.get(url)

z = zipfile.ZipFile(StringIO.StringIO(r.content))

df = pd.read_csv(z.open("ml-1m/ratings.dat"), sep='::', header=None)
df.columns = ["userId", "movieId", "rating", "timestamp"]

print(len(df))

def num_users_per_movie(group):
    group["num_users_per_movie"] = len(group)
    return group
df = df.groupby("movieId").apply(num_users_per_movie)
df = df[df["num_users_per_movie"] > 1]
del df["num_users_per_movie"]

print(len(df))

user_id_map = {user_id:i for i,user_id in enumerate(list(set(df['userId'])))}
movie_id_map = {movie_id:i for i,movie_id in enumerate(list(set(df['movieId'])))}

df['userId'] = df['userId'].apply(lambda u: user_id_map[u])
df['movieId'] = df['movieId'].apply(lambda m: movie_id_map[m])

# average_movies_per_user = min(df.groupby("userId").apply(len))
# average_users_per_movie = min(df.groupby("movieId").apply(len))
#
# print(average_movies_per_user,average_users_per_movie)

if os.path.isdir("data"):
    shutil.rmtree("data")

os.makedirs("data")

df.to_csv("data/ratings_med.csv", index=False)
