from __future__ import division
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.cross_validation import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LSHForest
import sys
from similarity import *
from combine import *
from metrics import *


def validation(users, movies, ratings, method, centrality_measure, alpha=0.5):
    seed = 470597
    kf = KFold(len(ratings), n_folds=10, shuffle=True, random_state=seed)
    for train, test in kf:
        # Split the ratings into train and test
        users_train, movies_train, ratings_train = users[train], movies[train], ratings[train]
        users_test, movies_test, ratings_test = users[test], movies[test], ratings[test]

        um_dense = train_model(users_train, movies_train, ratings_train, method=method, centrality_measure=centrality_measure, alpha=alpha)
        recs_df = serialize_recs(um_dense, users_train, movies_train)
        test_df = save_test_data(users_test, movies_test, ratings_test)

        precision_at_N,recall_at_N = precision_recall_at_N(recs_df, test_df, top_N=100)
        precision_threshold,recall_threshold = precision_recall_threshold(recs_df, test_df)
        ndcg = compute_ndcg(recs_df, test_df)
        rmse = compute_rmse(recs_df, test_df)

        return (method, centrality_measure, alpha, precision_at_N, recall_at_N, precision_threshold, recall_threshold, ndcg, rmse)


def train_model(users, movies, ratings, method, centrality_measure=None, alpha=.9):
    # Convert the train ratings into a User-Movie matrix
    um_sparse = convert_to_um_matrix(users, movies, ratings)

    # Complete the sparse UM matrix via the respective collaborative filtering algorithm

    # TODO: User CF is broken, all others seem to work as intended
    if method == 'user':
        similarity_matrix = compute_user_similarity(um_sparse).toarray()
        um_dense = user_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'movie':
        similarity_matrix = compute_movie_similarity(um_sparse).toarray()
        um_dense = item_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'user_centrality':
        similarity_matrix = compute_augmented_similarity(um_sparse, node_type='user', centrality_measure=centrality_measure, alpha=alpha)
        um_dense = user_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'movie_centrality':
        similarity_matrix = compute_augmented_similarity(um_sparse, node_type='movie', centrality_measure=centrality_measure, alpha=alpha)
        um_dense = item_based_recommendation_nnz(um_sparse, similarity_matrix)

    return um_dense


def serialize_recs(um_dense, users_train, movies_train):
    train_um_pairs = set(zip(users_train, movies_train))
    results_umr = [(um[0], um[1], rating) for um, rating in  np.ndenumerate(um_dense) if um not in train_um_pairs]

    recs_df = pd.DataFrame(results_umr, columns=['user', 'movie', 'predicted_rating'])
    recs_df = recs_df.groupby('user').apply(lambda x: x.sort_values('predicted_rating', ascending=False))

    recs_df.index = range(len(recs_df))
    recs_df.to_csv('./recs.csv', index=False)
    return recs_df


def save_test_data(users_test, movies_test, ratings_test):
    test_umr = zip(users_test, movies_test, ratings_test)
    test_df = pd.DataFrame(test_umr, columns=['user', 'movie', 'actual_rating'])
    test_df = test_df.groupby('user').apply(lambda x: x.sort_values('actual_rating', ascending=False))

    test_df.index = range(len(test_df))
    test_df.to_csv('./test.csv', index=False)
    return test_df


def compute_top_movies(um):
    averages = um.sum(0)/(um != 0).sum(0)
    return np.argsort(averages[0]).tolist()[::-1]


def user_based_recommendation_lsh(um, sim, ind, n_neighbors=6):
    sim = normalize(sim, axis=1, norm='l1')
    um_dense = np.vstack(tuple([sim[i].reshape(1,n_neighbors) * um[ind[i]] for i in range(len(ind))]))
    return um_dense


def user_based_recommendation(s_user, um):
    s_user = normalize(s_user, axis=1, norm='l1')
    um_dense = np.dot(s_user, um)
    return um_dense


def item_based_recommendation(um_sparse, s_movie):
    s_movie = normalize(s_movie, axis=0, norm='l1')
    um_dense = um_sparse * s_movie
    return um_dense


def user_based_recommendation_nnz(um_sparse, s_user):
    # Get nonzero indices of each movie vector
    nnz = [np.nonzero(um_sparse.T[i])[1] for i in range(um_sparse.shape[1])]

    # Compute the weighted average user_movie_rating*u_sim_weight for nonzero um ratings
    rows = []
    for k in range(um_sparse.shape[1]):
        row = um_sparse.T[k,nnz[k]] * normalize(s_user[nnz[k],:], axis=0, norm='l1')
        rows += [row.flatten()]
    um_dense = np.vstack(tuple(rows)).T
    return um_dense



# Item based recommendation with non zero ratings
def item_based_recommendation_nnz(um_sparse, s_movie):
    # Get nonzero indices of each user vector
    nnz = [np.nonzero(um_sparse[i])[1] for i in range(um_sparse.shape[0])]

    # Compute the weighted average user_movie_rating*m_sim_weight for nonzero um ratings
    rows = []
    for k in range(um_sparse.shape[0]):
        row = um_sparse[k,nnz[k]] * normalize(s_movie[nnz[k],:], axis=0, norm='l1')
        rows += [row.flatten()]
    um_dense = np.vstack(tuple(rows))
    return um_dense


if __name__ == "__main__":
    filename = './data/ratings_med.csv'
    users, movies, ratings = read_csv_data(filename)
    validation(users, movies, ratings, method='user', centrality_measure='', alpha=0.5)

    print("Done.")
