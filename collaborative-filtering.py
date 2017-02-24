from __future__ import division
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.cross_validation import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LSHForest 
import sys

user_id_map = dict()
movie_id_map = dict()

def read_csv_data(filename):
    ratings_df = pd.read_csv(filename)
    users = ratings_df["userId"].as_matrix()
    movies = ratings_df["movieId"].as_matrix()
    ratings = ratings_df["rating"].as_matrix()

    global user_id_map
    user_id_map = {user: i for i, user in enumerate(list(set(users)))}
    user_ind = np.array([user_id_map[user] for user in users])

    global movie_id_map
    movie_id_map = {movie: i for i, movie in enumerate(list(set(movies)))}
    movie_ind = np.array([movie_id_map[movie] for movie in movies])

    return (user_ind, movie_ind, ratings)


def read_dat_data(filename):
    ratings_df = pd.read_csv(filename, sep='::', header=None)
    ratings_df.columns = ["userId", "movieId", "rating", "timestamp"]
    users = ratings_df["userId"].as_matrix()
    movies = ratings_df["movieId"].as_matrix()
    ratings = ratings_df["rating"].as_matrix()

    global user_id_map
    user_id_map = {user: i for i, user in enumerate(list(set(users)))}
    user_ind = np.array([user_id_map[user] for user in users])

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


def hash_user_similarity(um):
    lsh = LSHForest()
    lsh.fit(um)
    dist, ind = lsh.kneighbors(um, n_neighbors=6, return_distance=True)
    sim = 1 - dist
    return sim, ind


def fold(users, movies, ratings):
    kf = KFold(len(users), n_folds=10, shuffle=True)
    for train, test in kf:
        # Split the ratings into train and test
        users_train, movies_train, ratings_train = users[train], movies[train], ratings[train]
        users_test, movies_test, ratings_test = users[test], movies[test], ratings[test]

        # Convert the train ratings into a User-Movie matrix
        um = convert_to_um_matrix(users_train, movies_train, ratings_train)

        # Compute user similarity via LSH and movie similarity via MMM
        sim, ind = hash_user_similarity(um)
        # s_movie = compute_movie_similarity(um)

        # Complete the sparse UM matrix via the collaborative filtering algorithm
        um_dense = fill_hash_matrix(um, sim, ind)

        # Compute metrics for the completed matrix with users that are in train and test
        intersect_users = set(users_test) & set(users_train)
        test_list = zip(users_test, movies_test, ratings_test)

        # Exclude the training ratings to avoid lookahead bias
        user_actual_list = [(user, [(m,r) for u,m,r in test_list if u==user]) for user in intersect_users]
        user_movie_train_pairs = set(zip(users_train, movies_train))

        # print(rmse(um_dense, user_actual_list))
        print(precision_at_N(um_dense, user_actual_list, user_movie_train_pairs))


def precision_at_N(um_dense, user_actual_list, user_movie_train_pairs, top_N=6):
    precisions = []
    for user, movie_ratings in user_actual_list:
        top_sorted_actual = sorted(movie_ratings, key=lambda x : x[1])[::-1][:top_N]

        # Sort the predictions by rating and filter out the movies that were in the training set 
        top_sorted_predicted = np.argsort(um_dense[user]).tolist()[::-1]
        top_sorted_predicted = [m for m in top_sorted_predicted if (user, m) not in user_movie_train_pairs][:top_N]
        
        overlap = len({m for m,r in top_sorted_actual} & set(top_sorted_predicted))
        total_rated = min(len(top_sorted_actual), len(top_sorted_predicted))
        if total_rated: precisions.append(overlap / total_rated)
    return np.mean(precisions)


def rmse(um_dense, user_actual_list):
    errors = []
    for user, movie_ratings in user_actual_list:
        for movie, actual_rating in movie_ratings:
            predicted_rating = um_dense[user][movie]
            errors.append((predicted_rating - actual_rating)**2)
    return np.sqrt(np.mean(errors))


def fill_hash_matrix(um, sim, ind, n_neighbors=6):
    um_dense = np.vstack(tuple([sim[i].reshape(1,n_neighbors) * um[ind[i]] for i in range(len(ind))]))
    s_sum = sim.sum(axis=1)
    return (um_dense.T / s_sum).T


def fill_matrix(s_user, um):
    um_dense = np.dot(s_user, um)
    s_sum = s_user.sum(axis=0)
    return (um_dense.T / s_sum).T


def compute_top_movies(um):
    averages = um.sum(0)/(um != 0).sum(0)
    return np.argsort(averages[0]).tolist()[::-1]


if __name__ == "__main__":
    filename = sys.argv[1]
    users, movies, ratings = read_csv_data(filename) if filename[-3:] == 'csv' else read_dat_data(filename)
    fold(users, movies, ratings)

    print("Done.")
