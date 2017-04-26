from __future__ import division
import numpy as np
import networkx as nx
import time
import pandas
from similarity import *
import sys

# TODO: Use larger matrices
def compute_centrality(sim, graph_type):
    G = nx.from_numpy_matrix(sim.toarray())
    centrality_functions = {"degree" : nx.degree_centrality,
                            "closeness" : nx.closeness_centrality,
                            "betweenness" : nx.betweenness_centrality,
                            "eigenvector" : nx.eigenvector_centrality
                            #"katz" : nx.katz_centrality}
                            }
    data = dict()
    for name, f in centrality_functions.items():
        print(graph_type, name)
        data[name] = f(G)
        df = pandas.DataFrame.from_dict(data)
        df.sort_index(inplace=True)
        df.to_csv("./centrality_data/{}_centrality.csv".format(graph_type), index=True)

    return df.as_matrix()

if __name__ == '__main__':
    filename = sys.argv[1]
    users, movies, ratings = read_csv_data(filename) if filename[-3:] == 'csv' else read_dat_data(filename)

    um = convert_to_um_matrix(users, movies, ratings)
    s_user = compute_user_similarity(um)
    s_movie = compute_movie_similarity(um)

    compute_centrality(s_user, 'user')
    compute_centrality(s_movie, 'movie')

