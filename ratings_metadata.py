from __future__ import division
import numpy as np 
import pandas as pd
import sys

filename = sys.argv[1]
ratings_df = pd.read_csv(filename)
n = len(set(ratings_df['movieId']))
m = len(set(ratings_df['userId']))
p = len(ratings_df)
avg_movies_per_user = np.mean(ratings_df.groupby('userId').apply(len))
avg_users_per_movie = np.mean(ratings_df.groupby('movieId').apply(len))

print "Number of users: {}".format(m)
print "Number of movies: {}".format(n)
print "Number of ratings: {}".format(p)
print "Density: {}%".format(round(p / (m*n)*100, 2))
print "Average users per movie: {}".format(round(avg_users_per_movie, 2))
print "Average movies per user: {}".format(round(avg_movies_per_user, 2))