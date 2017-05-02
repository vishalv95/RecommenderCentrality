import collaborative_filtering as cf
import similarity as s
import centrality as c
import os
import sys

# sim,ind = s.hash_user_similarity(s.convert_to_um_matrix(*s.read_csv_data("data/ratings_med.csv")))
# print("hash done")
# g = c.construct_graph(ind,sim)
# print("graph done")
# print(c.compute_centrality(g, "user"))
# print("centrality done")
#print(cf.validation(*s.read_csv_data("data/ratings_med.csv"), method="user", centrality_measure="betweenness", alpha=0))
#print(cf.validation(*s.read_csv_data("data/ratings_med.csv"), method="user", centrality_measure="betweenness", alpha=0))
method = sys.argv[1]
centrality = sys.argv[2]
alpha = float(sys.argv[3])
print(cf.validation(*s.read_csv_data("data/ratings_med.csv"), method=method, centrality_measure=centrality, alpha=alpha))
os.rename("recs.csv", "recs_{}_{}_{:3.1f}.csv".format(method, centrality, alpha))
#print(cf.validation(*s.read_csv_data("data/ratings_med.csv"), method="user_centrality", centrality_measure="betweenness", alpha=0))
