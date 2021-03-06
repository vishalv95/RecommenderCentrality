from __future__ import division
import numpy as np
import networkx as nx
import time
import pandas
from similarity import *
import sys
from particle_filtering import *

# TODO: Use larger matrices
def compute_centrality(sim, dis, graph_type):
    G = nx.from_numpy_matrix(sim.toarray())

    # Cosine similarity based centrality functions
    sim_centrality_functions = {"degree" : nx.degree_centrality,
                                "eigenvector" : nx.eigenvector_centrality}

    # Cosine distance based centrality functions
    dis_centrality_functions = {"betweenness" : nx.betweenness_centrality,
                                "closeness" : nx.closeness_centrality}

    data = dict()
    for name, f in sim_centrality_functions.items():
        print(graph_type, name)
        data[name] = f(G, max_iter=3000) if name == "eigenvector" and graph_type == "movie" else f(G)
        df = pandas.DataFrame.from_dict(data)
        df.sort_index(inplace=True)
        df.to_csv("./centrality_data/{}_centrality.csv".format(graph_type), index=True)

    for name, f in dis_centrality_functions.items():
        print(graph_type, name)
        data[name] = f(G)
        df = pandas.DataFrame.from_dict(data)
        df.sort_index(inplace=True)
        df.to_csv("./centrality_data/{}_centrality.csv".format(graph_type), index=True)

    data["particle_filtering"] = user_particle_filter('data/ratings_med.csv') if graph_type == "user" else movie_particle_filter('data/ratings_med.csv')

    df = pandas.DataFrame.from_dict(data)
    df.sort_index(inplace=True)
    df.fillna(value=0, inplace=True)
    df.to_csv("./centrality_data/{}_centrality.csv".format(graph_type), index=True)

    return df.as_matrix()

if __name__ == '__main__':
    filename = sys.argv[1]
    users, movies, ratings = read_csv_data(filename) if filename[-3:] == 'csv' else read_dat_data(filename)

    um = convert_to_um_matrix(users, movies, ratings)
    s_user = compute_user_similarity(um)
    s_movie = compute_movie_similarity(um)

    print(compute_centrality(s_user, 'user'))
    print(compute_centrality(s_movie, 'movie'))

