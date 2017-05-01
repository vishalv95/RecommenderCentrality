import pandas as pd
import numpy

centrality_df = pd.read_csv("centrality_data/user_centrality.csv", index_col=0)
print(centrality_df.corr())

print("\n\n")

centrality_df = pd.read_csv("centrality_data/movie_centrality.csv", index_col=0)
corr = centrality_df.corr()
print corr
corr.to_csv('./centrality_correlations.csv')
