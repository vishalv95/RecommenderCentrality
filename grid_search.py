import pandas as pd
import numpy as np
import centrality as c
import similarity as s
from collaborative_filtering import *

filename = "data/ratings_med.csv"

sim,ind = s.hash_user_similarity(s.convert_to_um_matrix(*s.read_csv_data(filename)))
dist = 1.0 - sim

g_sim = s.construct_graph(ind, sim)
g_dist = s.construct_graph(ind, dist)
c.compute_centrality(g_sim, g_dist, "user")

sim,ind = s.hash_movie_similarity(s.convert_to_um_matrix(*s.read_csv_data(filename)))
dist = 1.0 - sim
g_sim = s.construct_graph(ind, sim)
g_dist = s.construct_graph(ind, dist)
c.compute_centrality(g_sim, g_dist, "movie")

# compute for all

centrality_types = pd.read_csv("centrality_data/user_centrality.csv", index_col=0).columns
users,movies,ratings = read_csv_data("data/ratings_med.csv")
result_rows = []
cols = ["method", "centrality_measure", "alpha", "precision_at_N", "recall_at_N",
                "precision_threshold", "recall_threshold", "ndcg", "rmse", "auc_threshold"]

for method in ["movie_centrality", "user_centrality"]:
    for centrality_measure in centrality_types:
        for alpha in np.arange(0.0, 1.1, 0.1):
            result_rows += [validation(users, movies, ratings, method, centrality_measure, alpha)]
            print(result_rows[-1])
            df = pd.DataFrame(result_rows, columns=cols)
            df.to_csv("grid_search_results.csv", index=False)
