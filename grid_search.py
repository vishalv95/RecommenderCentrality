import pandas as pd
import numpy as np
from collaborative_filtering import *

centrality_types = pd.read_csv("centrality_data/user_centrality.csv", index_col=0).columns
users,movies,ratings = read_csv_data("data/ratings.csv")
result_rows = []
cols = ["method", "centrality_measure", "alpha", "precision_at_N", "recall_at_N",
		"precision_threshold", "recall_threshold", "ndcg", "rmse"]

for method in ["user_centrality", "movie_centrality"]:
	for centrality_measure in centrality_types:
		for alpha in np.arange(0, 1.1, 0.1):
			result_rows += [validation(users, movies, ratings, method, centrality_measure, alpha)]

			df = pd.DataFrame(result_rows, columns=cols)
			df.to_csv("grid_search_results.csv", index=False)