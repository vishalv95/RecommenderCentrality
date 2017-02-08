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
    frequency = read_data(sys.argv[1])
    plt.xlabel("Number of ratings")
    plt.ylabel("Number of users")
    plt.title("Distribution of number of ratings per user")
    plt.scatter(list(int(k) for k in frequency.keys()), list(frequency.values()), label="test", s=1)
    plt.yscale('log')
    plt.savefig(sys.argv[2])

