from __future__ import division
import numpy as np
import networkx as nx
import time
import pandas

def compute_centrality(sim):
    start = time.time()
    G = nx.from_numpy_matrix(sim.toarray())
    centrality_functions = {"degree" : nx.degree_centrality,
                            "closeness" : nx.closeness_centrality,
                            "betweenness" : nx.betweenness_centrality,
                            "eigenvector" : nx.eigenvector_centrality,
                            "katz" : nx.katz_centrality}

    data = {name: f(G) for name,f in centrality_functions.items()}
    df = pandas.DataFrame.from_dict(data)
    df.sort_index(inplace=True)
    df.to_csv("centrality.csv", index=True)

    return df.as_matrix()
