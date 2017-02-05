import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.cross_validation import KFold

def read_data(filename):
    ratings_df = pd.read_csv(filename)
    users = ratings_df["userId"].as_matrix()
    movies = ratings_df["movieId"].as_matrix()
    ratings = ratings_df["rating"].as_matrix()
    return (users, movies, ratings)

def convert_to_um_matrix(users, movies, ratings):
    um = csr_matrix((ratings, (users, movies)), shape=(len(set(users)), len(set(movies))))
    return um

def compute_user_similarity(um):
    um_norm = normalize(um, axis=0, norm="l2")
    return np.dot(um_norm, um_norm.T)

def compute_movie_similarity(um):
    um_norm = normalize(um, axis=1, norm="l2")
    return np.dot(um_norm.T, um_norm)

def fold(users, movies, ratings):
    kf = KFold(n_splits=10)
    for train, test in kf.split(users):
        users_train, movies_train, ratings_train = users[train], movies[train], ratings[train]

        users_test, movies_test, ratings_test = users[test], movies[test], ratings[test]

        test_rating_triples = zip(users_test, movies_test, ratings_test)
        um = convert_to_um_matrix(users_train, movies_train, ratings_train)
        user_sim = compute_user_similarity(um)
        movie_sim = compute_movie_similarity(um)

        movie_filter(movie_sim, users_test)

def movie_filter(movie_sim, users, n_rec=6):
    pass
