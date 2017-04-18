from __future__ import division
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize
from sklearn.cross_validation import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LSHForest
import sys
from similarity import *


def validation(users, movies, ratings):
    kf = KFold(len(users), n_folds=10, shuffle=True)
    for train, test in kf:
        # Split the ratings into train and test
        users_train, movies_train, ratings_train = users[train], movies[train], ratings[train]
        users_test, movies_test, ratings_test = users[test], movies[test], ratings[test]

        # Convert the train ratings into a User-Movie matrix
        um = convert_to_um_matrix(users_train, movies_train, ratings_train)

        # Compute user similarity via LSH and movie similarity via MMM
        sim, ind = hash_user_similarity(um)
        s_movie = compute_movie_similarity(um)

        # Complete the sparse UM matrix via the collaborative filtering algorithm
        um_dense = item_based_recommendation_nnz(um, s_movie)
        # um_dense = fill_hash_matrix(um, sim, ind)

        # Compute metrics for the completed matrix with users that are in train and test
        intersect_users = set(users_test) & set(users_train)
        test_list = zip(users_test, movies_test, ratings_test)

        # Exclude the training ratings to avoid lookahead bias
        user_mr_test = [(user, [(m,r) for u,m,r in test_list if u==user]) for user in intersect_users]
        
        print("RMSE: " + str(rmse(um_dense, user_mr_test)))
        print("Precision @ N: " + str(precision_at_N(um_dense, user_mr_test)))


def precision_at_N(um_dense, user_mr_test, top_N=6):
    precisions = []
    
    for user, movie_ratings in user_mr_test:
        # Compute the top movies in the test set for a user, and the ordering of those movies by model prediction
        sorted_test_movies_actual = np.array([movie for movie, rating in sorted(movie_ratings, key=lambda x : x[1], reverse=True)])
        sorted_test_movies_predict = np.argsort(um_dense[user,:])[::-1]

        top_test_movies_actual = sorted_test_movies_actual[:top_N]
        top_test_movies_predict = sorted_test_movies_predict[np.in1d(sorted_test_movies_predict, sorted_test_movies_actual)][:top_N]

        # Compute % overlap in top N movies between actual and predicted as precision @ N
        overlap = len(set(top_test_movies_actual) & set(top_test_movies_predict))
        total_rated = min(len(top_test_movies_actual), len(top_test_movies_predict))
        if total_rated: precisions.append(overlap / total_rated)

    return np.mean(precisions)

# TODO: Other metrics
def ndcg(um_dense, user_mr_test, top_N=6):
    pass


def map(um_dense, user_mr_test, top_N=6):
    pass

# Compute the root mean squared error between a prediction and an actual test rating
def rmse(um_dense, user_mr_test):
    errors = []
    for user, movie_ratings in user_mr_test:
        for movie, actual_rating in movie_ratings:
            predicted_rating = um_dense[user, movie]
            errors.append((predicted_rating - actual_rating)**2)
    return np.sqrt(np.mean(errors))


def fill_hash_matrix(um, sim, ind, n_neighbors=6):
    sim = normalize(sim, axis=1, norm='l1')
    um_dense = np.vstack(tuple([sim[i].reshape(1,n_neighbors) * um[ind[i]] for i in range(len(ind))]))
    return um_dense


def fill_matrix(s_user, um):
    normalize(s_user, axis=1, norm='l1')
    um_dense = np.dot(s_user, um)
    return um_dense


def compute_top_movies(um):
    averages = um.sum(0)/(um != 0).sum(0)
    return np.argsort(averages[0]).tolist()[::-1]


def item_based_recommendation(um_sparse, s_movie):
    s_movie = normalize(s_movie, axis=0, norm='l1')
    um_dense = um_sparse * s_movie
    return um_dense


# Item based recommendation with non zero ratings
def item_based_recommendation_nnz(um_sparse, s_movie):
    # Get nonzero indices of each user vector
    nnz = [np.nonzero(um_sparse[i])[1] for i in range(um_sparse.shape[0])]

    # Compute the weighted average user_movie_rating*m_sim_weight for nonzero um ratings
    rows = []
    for k in range(um_sparse.shape[0]):
        row = um_sparse[k,nnz[k]] * normalize(s_movie[nnz[k],:], axis=0, norm='l1')
        rows += [row.toarray().flatten()]
    um_dense = np.vstack(tuple(rows))
    return um_dense


# Construct the graph from an index and similarity matrix 
def construct_graph(ind, sim):
    num_vertices = ind.shape[0]
    num_neighbors = ind.shape[1]
    coordinates = [(i, ind[i][j], sim[i][j]) for i in range(num_vertices) for j in range(num_neighbors)]
    i,j,data = zip(*coordinates)
    return coo_matrix((data, (i,j)), shape=(num_vertices, num_vertices)).tocsr()


if __name__ == "__main__":
    filename = sys.argv[1]
    users, movies, ratings = read_csv_data(filename) if filename[-3:] == 'csv' else read_dat_data(filename)
    validation(users, movies, ratings)

    print("Done.")
