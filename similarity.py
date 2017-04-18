from __future__ import division
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize
from sklearn.cross_validation import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LSHForest
import sys

user_id_map = dict()
movie_id_map = dict()

def read_csv_data(filename):
    # Read users, movies, ratings as numpy arrays
    ratings_df = pd.read_csv(filename)
    users = ratings_df["userId"].as_matrix()
    movies = ratings_df["movieId"].as_matrix()
    ratings = ratings_df["rating"].as_matrix()

    # Map user ids to indices
    global user_id_map
    user_id_map = {user: i for i, user in enumerate(list(set(users)))}
    user_ind = np.array([user_id_map[user] for user in users])

    # Map movie ids to indices
    global movie_id_map
    movie_id_map = {movie: i for i, movie in enumerate(list(set(movies)))}
    movie_ind = np.array([movie_id_map[movie] for movie in movies])

    return (user_ind, movie_ind, ratings)


def read_dat_data(filename):
    # Read users, movies, ratings as numpy arrays
    ratings_df = pd.read_csv(filename, sep='::', header=None)
    ratings_df.columns = ["userId", "movieId", "rating", "timestamp"]
    users = ratings_df["userId"].as_matrix()
    movies = ratings_df["movieId"].as_matrix()
    ratings = ratings_df["rating"].as_matrix()

    # Map user ids to indices
    global user_id_map
    user_id_map = {user: i for i, user in enumerate(list(set(users)))}
    user_ind = np.array([user_id_map[user] for user in users])

    # Map movie ids to indices
    global movie_id_map
    movie_id_map = {movie: i for i, movie in enumerate(list(set(movies)))}
    movie_ind = np.array([movie_id_map[movie] for movie in movies])

    return (user_ind, movie_ind, ratings)


def convert_to_um_matrix(users, movies, ratings):
    um = csr_matrix((ratings, (users, movies)), shape=(len(user_id_map), len(movie_id_map)))
    return um


def compute_user_similarity(um):
    s_user = cosine_similarity(um, um, dense_output=False)
    s_user.setdiag(0)
    return s_user


def compute_movie_similarity(um):
    s_movie = cosine_similarity(um.T, um.T, dense_output=False)
    s_movie.setdiag(0)
    return s_movie


def hash_user_similarity(um, num_neighbors=6):
    lsh = LSHForest()
    lsh.fit(um)

    # Don't compare to self, remove first column, call 7 neighbors
    dist, ind = lsh.kneighbors(um, n_neighbors=num_neighbors+1, return_distance=True)
    sim = 1 - dist
    return sim[:,1:], ind[:,1:]

# TODO: Convert the hash scores to an adjacency matrix usable for centrality calculations
def hash_to_similarity(sim, ind):
    pass

