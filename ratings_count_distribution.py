import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

def count_ratings_per_user(filename):
    ratings_df = pd.read_csv(filename)
    user_counts = ratings_df.groupby('userId').apply(lambda x: len(x))
    user_counts.hist(bins=20)
    plt.xticks(np.arange(0, 2500, 200))
    plt.xlabel("Movies Rated by a User")
    plt.ylabel("Frequency")
    plt.title("Ratings per User Distribution")
    plt.show()


def count_users_per_rating(filename):
    ratings_df = pd.read_csv(filename)
    rating_counts = ratings_df.groupby('rating').apply(lambda x: len(x))
    rating_dist = rating_counts / np.sum(rating_counts)

    fig1, ax1 = plt.subplots()
    ax1.pie(rating_dist, labels=range(1,6),  autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')

    plt.show()


if __name__ == "__main__":
    # frequency = list(int(k) for k in read_data(sys.argv[1]).keys())
    count_users_per_rating(sys.argv[1])
#     plt.xlabel("Number of ratings")
#     plt.ylabel("Number of users")
#     plt.title("Distribution of number of ratings per user")
#     plt.hist(frequency, bins=100, edgecolor='k', linewidth=1, facecolor='w')
# #    plt.yscale('log')
#     plt.savefig(sys.argv[2])

