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
from popular import *


def validation(users, movies, ratings, method, centrality_measure=None, alpha=0.0, tfidf=False):
    seed = 470597
    kf = KFold(len(ratings), n_folds=10, shuffle=True, random_state=seed)
    for train, test in kf:
        # Split the ratings into train and test
        users_train, movies_train, ratings_train = users[train], movies[train], ratings[train]
        users_test, movies_test, ratings_test = users[test], movies[test], ratings[test]

        um_dense = train_model(users_train, movies_train, ratings_train, method=method, centrality_measure=centrality_measure, alpha=alpha, tfidf=tfidf)
        recs_df = serialize_recs(um_dense, users_train, movies_train, users_test, movies_test)
        test_df = save_test_data(users_test, movies_test, ratings_test)

        precision_top_N,recall_top_N,accuracy_top_N = classification_report_top_N(recs_df, test_df)
        precision_thresh,recall_thresh,accuracy_thresh = classification_report_thresh(recs_df, test_df)
        ndcg = compute_ndcg(recs_df, test_df)
        rmse = compute_rmse(recs_df, test_df)

        # Return after the first fold
        return (method, centrality_measure, alpha, precision_top_N, recall_top_N, accuracy_top_N, precision_thresh, recall_thresh, accuracy_thresh, ndcg, rmse)


def train_model(users, movies, ratings, method, centrality_measure=None, alpha=0.0, tfidf=False):
    # Convert the train ratings into a User-Movie matrix
    um_sparse = convert_to_um_matrix(users, movies, ratings, tfidf=tfidf)

    # Complete the sparse UM matrix via the respective collaborative filtering algorithm
    if method == 'user':
        similarity_matrix = compute_user_similarity(um_sparse).toarray()
        um_dense = user_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'movie':
        similarity_matrix = compute_movie_similarity(um_sparse).toarray()
        um_dense = item_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'user_lsh':
        sim, ind = hash_user_similarity(um_sparse)
        similarity_matrix = construct_graph(ind, sim).toarray()
        um_dense = user_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'movie_lsh':
        sim, ind = hash_movie_similarity(um_sparse)
        similarity_matrix = construct_graph(ind, sim).toarray()
        um_dense = item_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'user_lsh_smoothing':
        sim, ind = hash_user_similarity(um_sparse)
        similarity_matrix = construct_smooth_sim(ind, sim)
        um_dense = user_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'movie_lsh_smoothing':
        sim, ind = hash_movie_similarity(um_sparse)
        similarity_matrix = construct_smooth_sim(ind, sim)
        um_dense = item_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'user_centrality':
        similarity_matrix = compute_augmented_similarity(um_sparse, node_type='user', centrality_measure=centrality_measure, alpha=alpha)
        um_dense = user_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'movie_centrality':
        similarity_matrix = compute_augmented_similarity(um_sparse, node_type='movie', centrality_measure=centrality_measure, alpha=alpha)
        um_dense = item_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'user_centrality_lsh':
        similarity_matrix = compute_augmented_similarity_lsh(um_sparse, node_type='user', centrality_measure=centrality_measure, alpha=alpha)
        um_dense = user_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'movie_centrality_lsh':
        similarity_matrix = compute_augmented_similarity_lsh(um_sparse, node_type='movie', centrality_measure=centrality_measure, alpha=alpha)
        um_dense = item_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'popular':
        ratings_df = pd.read_csv('./data/ratings_med.csv')
        um_dense = popular_matrix(ratings_df)

    return um_dense


def serialize_recs(um_dense, users_train, movies_train, users_test, movies_test):
    user_set = set(users_train) & set(users_test)
    movie_set = set(movies_train) & set(movies_test)

    train_um_pairs = set(zip(users_train, movies_train))
    results_umr = [(um[0], um[1], rating) for um, rating in  np.ndenumerate(um_dense)
                    if um not in train_um_pairs and um[0] in user_set and um[1] in movie_set]

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


def save_train_data(users_train, movies_train, ratings_train):
    train_umr = zip(users_train, movies_train, ratings_train)
    train_df = pd.DataFrame(train_umr, columns=['user', 'movie', 'actual_rating'])
    train_df = train_df.groupby('user').apply(lambda x: x.sort_values('actual_rating', ascending=False))

    train_df.index = range(len(train_df))
    train_df.to_csv('./train.csv', index=False)
    return train_df


def compute_top_movies(um_sparse):
    averages = um_sparse.sum(0)/(um_sparse != 0).sum(0)
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
    # print validation(users, movies, ratings, method='user_lsh')
    print validation(users, movies, ratings, method='popular')
