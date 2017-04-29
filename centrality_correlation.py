import pandas as pd
import numpy

centrality_df = pd.read_csv("centrality_data/user_centrality.csv")
print(centrality_df.corr())

print("\n\n")

centrality_df = pd.read_csv("centrality_data/movie_centrality.csv")
print(centrality_df.corr())
