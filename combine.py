import pandas as pd 
import numpy as np 
from sklearn.preprocessing import normalize
from similarity import *

def load_centrality(centrality_file, node_type, centrality_measure):
	centrality_df = pd.read_csv(centrality_file)
	centrality_array = centrality_df[centrality_measure].as_matrix()
	return centrality_array


def load_similarity(ratings_file, node_type):
	users, movies, ratings = read_csv_data(ratings_file) if ratings_file[-3:] == 'csv' else read_dat_data(ratings_file)
	um = convert_to_um_matrix(users, movies, ratings)
	sim = compute_user_similarity(um) if node_type == 'user' else compute_movie_similarity(um)
	return sim
	

def combine(similarity_matrix, centrality_array, alpha=0.1):
	# Add weighted centrality score to each row in similarity matrix

	# TODO: Experiment with where we do the normalization step, before/after combine
	new_sim = np.apply_along_axis(lambda row: alpha*row + (1-alpha)*centrality_array, axis=1, arr=similarity_matrix)
	return new_sim


if __name__ == '__main__':
	centrality_array = load_centrality('./centrality_data/user_centrality.csv', 'user', 'degree')
	sim = load_similarity('./data/ratings.csv', 'user')
	new_sim = combine(sim.toarray(), centrality_array)
	