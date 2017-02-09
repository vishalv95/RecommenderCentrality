import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

def read_data(filename):
    ratings_df = pd.read_csv(filename)
    users = ratings_df["userId"].as_matrix()
    ratings = dict()
    frequency = dict()
    for user in users:
        if user in ratings:
            ratings[user] = ratings[user] + 1
        else:
            ratings[user] = 1
    for freq in list(ratings.values()):
        if freq in frequency:
            frequency[freq] = frequency[freq] + 1
        else:
            frequency[freq] = 1
    return frequency

if __name__ == "__main__":
    frequency = list(int(k) for k in read_data(sys.argv[1]).keys())
    plt.xlabel("Number of ratings")
    plt.ylabel("Number of users")
    plt.title("Distribution of number of ratings per user")
    plt.hist(frequency, bins=100, edgecolor='k', linewidth=1, facecolor='w')
#    plt.yscale('log')
    plt.savefig(sys.argv[2])

