from __future__ import division 
import pandas as pd 
import numpy as np 
from similarity import *

# Complete Graph 
users,movies,ratings = read_csv_data("data/ratings_med.csv")
um_sparse = convert_to_um_matrix(users, movies, ratings)

S_user = compute_user_similarity(um_sparse).toarray()
S_movie = compute_movie_similarity(um_sparse).toarray()

# Approximate kNN Graph
usim, uind = hash_user_similarity(um_sparse)
S_user_hash = construct_graph(uind, usim).toarray()

msim, mind = hash_movie_similarity(um_sparse)
S_movie_hash = construct_graph(mind, msim).toarray()

rows = []
for g in [S_user, S_movie, S_user_hash, S_movie_hash]:
	row = dict()

	# Unweighted Average Graph Degree
	degrees = (g != 0).sum(axis=0)
	row['MovieLens Average Node Degree'] = np.mean(degrees)

	# Weighted Average Graph Degree
	weighted_degrees = g.sum(axis=0)
	row['MovieLens Weighted Average Node Degree'] = np.mean(weighted_degrees)

	# Get edge cardinality by handshaking lemma
	E = np.sum(degrees) / 2
	# E = np.count_nonzero(g) / 2
	row['MovieLens |E|'] = E

	# Compute density with max |E|
	graph_cardinality = g.shape[0]
	density = (2*E) / (graph_cardinality * (graph_cardinality-1))
	row['MovieLens Graph Density'] = density

	rows += [row]

df = pd.DataFrame(rows)
df.index = ['S_user', 'S_movie', 'S_user_hash', 'S_movie_hash']
df.to_csv('./ml_graph_properties.csv')
print df