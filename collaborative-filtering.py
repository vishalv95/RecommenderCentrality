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


def validation_old(users, movies, ratings):
    kf = KFold(len(ratings), n_folds=10, shuffle=True)
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
        # um_dense = complete_hash_matrix(um, sim, ind)

        # Compute metrics for the completed matrix with users that are in train and test
        intersect_users = set(users_test) & set(users_train)
        test_list = zip(users_test, movies_test, ratings_test)

        # Exclude the training ratings to avoid lookahead bias
        user_mr_test = [(user, [(m,r) for u,m,r in test_list if u==user]) for user in intersect_users]

        print("Precision @ N: %.3f" % (precision_at_N(um_dense, user_mr_test),))
        print("Recall: %.3f" % (recall_at_N(um_dense, user_mr_test),))
        print("RMSE: %.3f" % (rmse(um_dense, user_mr_test),))


def validation(users, movies, ratings):
    kf = KFold(len(ratings), n_folds=10, shuffle=True)
    for train, test in kf:
        # Split the ratings into train and test
        users_train, movies_train, ratings_train = users[train], movies[train], ratings[train]
        users_test, movies_test, ratings_test = users[test], movies[test], ratings[test]

        um_dense = train_model(users_train, movies_train, ratings_train, method='user_centrality', centrality_measure='degree')
        results_df = serialize_results(um_dense, users_train, movies_train)
        test_df = save_test_data(users_test, movies_test, ratings_test)
        assert False

def train_model(users, movies, ratings, method, centrality_measure=None, alpha=.9):
    # Convert the train ratings into a User-Movie matrix
    um_sparse = convert_to_um_matrix(users, movies, ratings)

    # Complete the sparse UM matrix via the respective collaborative filtering algorithm

    # TODO: User CF is broken, all others seem to work as intended
    if method == 'user':
        similarity_matrix = compute_user_similarity(um_sparse)
        um_dense = user_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'movie':
        similarity_matrix = compute_movie_similarity(um_sparse)
        um_dense = item_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'user_centrality':
        similarity_matrix = compute_augmented_similarity(um_sparse, node_type='user', centrality_measure=centrality_measure, alpha=.9)
        um_dense = user_based_recommendation_nnz(um_sparse, similarity_matrix)

    elif method == 'movie_centrality':
        similarity_matrix = compute_augmented_similarity(um_sparse, node_type='movie', centrality_measure=centrality_measure, alpha=.9)
        um_dense = item_based_recommendation_nnz(um_sparse, similarity_matrix)

    return um_dense


def serialize_results(um_dense, users_train, movies_train):
    train_um_pairs = set(zip(users_train, movies_train))
    results_umr = [(um[0], um[1], rating) for um, rating in  np.ndenumerate(um_dense) if um not in train_um_pairs]

    results_df = pd.DataFrame(results_umr, columns=['user', 'movie', 'predicted_rating'])
    results_df = results_df.groupby('user').apply(lambda x: x.sort_values('predicted_rating', ascending=False))
    
    results_df.index = range(len(results_df))
    results_df.to_csv('./results.csv', index=False)
    return results_df


def save_test_data(users_test, movies_test, ratings_test):
    test_umr = zip(users_test, movies_test, ratings_test)
    test_df = pd.DataFrame(test_umr, columns=['user', 'movie', 'actual_rating'])
    test_df = test_df.groupby('user').apply(lambda x: x.sort_values('actual_rating', ascending=False))

    test_df.index = range(len(test_df))
    test_df.to_csv('./test.csv', index=False)
    return test_df


# TODO: Ensure that groups retain order
def precision_recall_at_N(results_df, test_df, top_N=6):
    results_df = results_df.groupby('user').apply(lambda x: x.head(top_N))
    test_df = test_df.groupby('user').apply(lambda x: x.head(top_N))
    overlap_df = results_df.merge(test_df, how='inner', on=['user', 'movie'])

    # TODO: save lengths instead to save memory
    precision = len(overlap_df) / len(results_df)
    recall = len(overlap_df) / len(test_df)
    return precision, recall


def precision_recall_threshold(results_df, test_df, thresh=3.0):
    results_df = results_df[results_df['predicted_rating'] >= thresh]
    test_df = test_df[test_df['actual_rating'] >= thresh]
    overlap_df = results_df.merge(test_df, how='inner', on=['user', 'movie'])
    
    # TODO: save lengths instead to save memory
    precision = len(overlap_df) / len(results_df)
    recall = len(overlap_df) / len(test_df)
    return precision, recall


def ndcg(results_df, test_df, thresh=3.0):
    def rank(df):
        df['rank'] = range(1,len(df)+1)

    results_df = results_df[results_df['predicted_rating'] >= thresh]
    results_df = results_df.groupby('user').apply(rank)

    test_df = test_df[test_df['actual_rating'] >= thresh]
    test_df = test_df.groupby('user').apply(rank)
    overlap_df = results_df.merge(test_df, how='inner', on=['user', 'movie'], suffixes=('_predicted', '_actual'))

    dcg = overlap_df.groupby('user').apply(lambda x: np.sum(x['predicted_rating'] / np.log2(x['rank_predicted'] + 1)))
    idcg = overlap_df.groupby('user').apply(lambda x: np.sum(x['actual_rating'] / np.log2(x['rank_actual'] + 1)))
    return np.mean(dcg / idcg)


def rmse_df(results_df, test_df):
    overlap_df = results_df.merge(test_df, how='inner', on=['user', 'movie'])
    rmse = np.sqrt(np.mean((overlap_df['predicted_rating'] - overlap_df['actual_rating']) ** 2))
    return rmse

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


def recall_at_N(um_dense, user_mr_test, top_N=6):
    recalls = []

    for user, movie_ratings in user_mr_test:
        # Compute "relevant" test movies as those with > 3.5 rating
        top_test_movies_actual = [movie for movie, rating in sorted(movie_ratings, key=lambda x : x[1], reverse=True) if rating >= 3]
        sorted_test_movies_predict = np.argsort(um_dense[user,:])[::-1]
        top_test_movies_predict = sorted_test_movies_predict[np.in1d(sorted_test_movies_predict, top_test_movies_actual)][:len(top_test_movies_actual)]

        # Compute % overlap in top N movies between actual and predicted as recall @ N
        overlap = len(set(top_test_movies_actual) & set(top_test_movies_predict))
        num_relevant = min(len(top_test_movies_actual), len(top_test_movies_predict))
        if num_relevant: recalls.append(overlap / num_relevant)

    return np.mean(recalls)


# Compute the root mean squared error between a prediction and an actual test rating
def rmse(um_dense, user_mr_test):
    errors = []
    for user, movie_ratings in user_mr_test:
        for movie, actual_rating in movie_ratings:
            predicted_rating = um_dense[user, movie]
            errors.append((predicted_rating - actual_rating)**2)
    return np.sqrt(np.mean(errors))


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
        rows += [row.toarray().flatten()]
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
        rows += [row.toarray().flatten()]
    um_dense = np.vstack(tuple(rows))
    return um_dense

if __name__ == "__main__":
    filename = sys.argv[1]
    users, movies, ratings = read_csv_data(filename) if filename[-3:] == 'csv' else read_dat_data(filename)
    validation(users, movies, ratings)

    print("Done.")
