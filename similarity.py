from __future__ import division
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize
from sklearn.cross_validation import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LSHForest
import sys

def read_csv_data(filename):
    # Read users, movies, ratings as numpy arrays
    ratings_df = pd.read_csv(filename)
    users = ratings_df["userId"].as_matrix()
    movies = ratings_df["movieId"].as_matrix()
    ratings = ratings_df["rating"].as_matrix()

    return (users, movies, ratings)


def convert_to_um_matrix(users, movies, ratings):
    um = csr_matrix((ratings, (users, movies)))
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


def hash_movie_similarity(um, num_neighbors=6):
    lsh = LSHForest()
    lsh.fit(um.T)

    # Don't compare to self, remove first column, call 7 neighbors
    dist, ind = lsh.kneighbors(um.T, n_neighbors=num_neighbors+1, return_distance=True)
    sim = 1 - dist
    return sim[:,1:], ind[:,1:]


# Construct the graph from an index and similarity matrix
def construct_graph(ind, weight):
    degree = ind.shape[0]
    num_neighbors = ind.shape[1]
    coordinates = [(i, ind[i][j], weight[i][j]) for i in range(degree) for j in range(num_neighbors)]
    i,j,data = zip(*coordinates)
    return csr_matrix((data+data, (i+j,j+i)), shape=(degree, degree))

