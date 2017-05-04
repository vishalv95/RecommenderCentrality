
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import pysparnn.cluster_index as ci
import numpy as np 
from similarity import *

# TODO: Use cosine distance 
def kmeans_cluster_um(um, knn, k_clusters, cf_type='user'):
	if cf_type == 'movie': um = um.T
	# Normalize for Euclidean Distance = Cosine Distance
	um = normalize(um, axis=1)
	seed = 470597
	km = KMeans(n_clusters=k_clusters, init='random', random_state=seed)
	clusters = km.fit_predict(um)

	coords = []
	for i in range(k_clusters):
		cluster_assign = np.where(clusters == i)[0]
		indexes = {cluster_i: actual_i for cluster_i, actual_i in enumerate(cluster_assign)}
		similarities = cosine_similarity(um[cluster_assign])
		coords += sort_similarities(similarities, indexes, knn+1)
	i,j,data = zip(*coords)
	sim = csr_matrix((data+data,(i+j,j+i)), shape=(um.shape[0], um.shape[0])).toarray()
	np.fill_diagonal(sim, 0)
	return sim


def sort_similarities(similarities, indexes, knn):
	coords = []
	for cluster_i, actual_i in indexes.items():
		row = similarities[cluster_i,:]
		neighbor_cis = list(np.argsort(row)[::-1][:knn])
		neighbors_sim = sorted(row)[::-1][:knn]
		coords += [(actual_i, indexes[neighbor_cis[i]], neighbors_sim[i]) for i in range(knn)]
	return coords


# Cluster Pruning 
def pysparnn_cluster_um(um, knn, k_clusters=8, cf_type='user'):
	if cf_type == 'movie': um = um.T
	num_vectors = um.shape[0]

	k_clusters = int(np.sqrt(num_vectors))
	return_data = range(num_vectors)
	cp = ci.MultiClusterIndex(um, return_data)
	results = cp.search(um, k=knn+1, k_clusters=k_clusters, return_distance=True)

	dist, ind = results_to_matrices(results)
	sim = 1.0 - dist
	return sim[:,1:], ind[:,1:]


def results_to_matrices(results):
	ind, dist = [], []
	for row in results:
		distance, index = zip(*row)
		dist += [distance]
		ind += [index]
	dist = np.array(dist)
	ind = np.array(ind).astype(int)
	return dist, ind


if __name__ == '__main__':
	users, movies, ratings = read_csv_data('./data/ratings_med.csv')
	um_sparse = convert_to_um_matrix(users, movies, ratings, tfidf=True)
	# print pysparnn_cluster_um(um_sparse, 6)
	print sum(kmeans_cluster_um(um_sparse, 6, 15).sum(0) - kmeans_cluster_um(um_sparse, 6, 15).sum(1))