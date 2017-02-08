import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.cross_validation import KFold
from sklearn.metrics.pairwise import cosine_similarity
import sys

user_id_map = dict()
movie_id_map = dict()

def read_data(filename):
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

def convert_to_um_matrix(users, movies, ratings):
#    print(str(len(set(users))))
#    print(str(len(set(movies))))

    um = csr_matrix((ratings, (users, movies)), shape=(len(user_id_map), len(movie_id_map)))
    return um

def compute_user_similarity(um):
    s_user = cosine_similarity(um, um, dense_output=False)
#    assert all(s_user == s_user.T)
    #print(s_user[:5][:5])
    s_user.setdiag(0)
    return s_user

def compute_movie_similarity(um):
    s_movie = cosine_similarity(um.T, um.T, dense_output=False)
#    print(s_movie[:5][:5])
    s_movie.setdiag(0)
    return s_movie

def fold(users, movies, ratings):
    kf = KFold(len(users), n_folds=10, shuffle=True)
    triple_list = list(zip(users, movies, ratings))
    for train, test in kf:
        users_train, movies_train, ratings_train = users[train], movies[train], ratings[train]

        users_test, movies_test, ratings_test = users[test], movies[test], ratings[test]

        um = convert_to_um_matrix(users_train, movies_train, ratings_train)
        s_user = compute_user_similarity(um)
        s_movie = compute_movie_similarity(um)

        um_dense = fill_matrix(s_user, um)

        intersect_user_list = set(users_test) & set(users_train)
        user_actual_list = []
        user_set = {u for u,m,r in triple_list}
        for user in intersect_user_list:
            mov_rating = [(m, r) for u, m, r in triple_list if u == user]
            if mov_rating:
                user_actual_list.append((user, mov_rating))

        print(precision_at_N(um_dense, user_actual_list))


def precision_at_N(um_dense, user_actual_list, top_N=6):
    precisions = []
    for user, movie_ratings in user_actual_list:
        top_sorted_actual = sorted(movie_ratings, key=lambda x : x[1])[::-1][:top_N]
#        print(um_dense[user].shape)
        top_sorted_predicted = np.argsort(um_dense[user]).tolist()[0][::-1][:top_N]
#        print(len(top_sorted_predicted[0]))
#        print(top_sorted_actual[:5])
#        print(top_sorted_predicted[:5])
        overlap = len({m for m,r in top_sorted_actual} & set(top_sorted_predicted))
        total_rated = min(len(top_sorted_actual), len(top_sorted_predicted))
        precisions.append(overlap / total_rated)
    return np.mean(precisions)

def fill_matrix(s_user, um):
    um_dense = np.dot(s_user, um)
    s_sum = s_user.sum(axis=0)
    return (um_dense.T / s_sum).T
#    return um_dense
#    return np.divide(um_dense, s_sum)

if __name__ == "__main__":
    users, movies, ratings = read_data(sys.argv[1])
    fold(users, movies, ratings)
#    um = convert_to_um_matrix(users, movies, ratings)
#    s_user = compute_user_similarity(um)
#    um_dense = fill_matrix(s_user, um)
#    print(um_dense[:5][:5])

    print("Done.")
