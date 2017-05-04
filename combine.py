import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from similarity import *

def load_centrality(node_type, centrality_measure):
	centrality_file = './centrality_data/{}_centrality.csv'.format(node_type)
	centrality_df = pd.read_csv(centrality_file)
	centrality_array = centrality_df[centrality_measure].as_matrix()
	return centrality_array


def load_similarity(ratings_file, node_type):
	users, movies, ratings = read_csv_data(ratings_file) if ratings_file[-3:] == 'csv' else read_dat_data(ratings_file)
	um = convert_to_um_matrix(users, movies, ratings)
	sim = compute_user_similarity(um) if node_type == 'user' else compute_movie_similarity(um)
	return sim


# Axis should 1 for row (user case), 0 for col (item case)
def compute_augmented_similarity(um_sparse, node_type, centrality_measure, alpha=.9):
	centrality_array = load_centrality(node_type, centrality_measure)
	centrality_array = normalize(centrality_array, norm='l1').flatten()

	axis = 1 if node_type == 'user' else 0
	similarity_matrix = compute_user_similarity(um_sparse) if node_type == 'user' else compute_movie_similarity(um_sparse)
	similarity_matrix = similarity_matrix.toarray()
	similarity_matrix = normalize(similarity_matrix, norm='l1', axis=axis)

	augmented_similarity = np.apply_along_axis(lambda vec: (1-alpha)*vec + alpha*centrality_array, axis=axis, arr=similarity_matrix)
	return augmented_similarity


# Use sparse LSH similarity matrix as basis for augmentation
def compute_augmented_similarity_lsh(um_sparse, node_type, centrality_measure, alpha=.9):
	centrality_array = load_centrality(node_type, centrality_measure)
	centrality_array = normalize(centrality_array, norm='l1').flatten()

	axis = 1 if node_type == 'user' else 0
	sim, ind = hash_user_similarity(um_sparse) if node_type=='user' else hash_movie_similarity(um_sparse)
	similarity_matrix = construct_graph(ind, sim)
	similarity_matrix = similarity_matrix.toarray()
	similarity_matrix = normalize(similarity_matrix, norm='l1', axis=axis)

	augmented_similarity = np.apply_along_axis(lambda vec: alpha*vec + (1-alpha)*centrality_array, axis=axis, arr=similarity_matrix)
	return augmented_similarity


def influence_matrix(centrality_measure):
	cu = load_centrality('user', centrality_measure)
	ci = load_centrality('item', centrality_measure)
	F = np.dot(ci, cu)
	return F


if __name__ == '__main__':
	# users, movies, ratings = read_csv_data('./data/ratings.csv')
	# um_sparse = convert_to_um_matrix(users, movies, ratings)

	# augmented_similarity = compute_augmented_similarity(um_sparse, node_type='user', centrality_measure='eigenvector')
	print influence_matrix('eigenvector')
