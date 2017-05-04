import pandas as pd 
import numpy as np 
from scipy.sparse import csr_matrix

def get_popular_movies_df(ratings_df):
	counts = ratings_df.groupby('movieId').apply(lambda x: len(x))
	avg_rating = ratings_df.groupby('movieId').apply(lambda x: np.mean(x['rating']))
	df = pd.DataFrame()
	df['counts'] = counts
	df['avg_rating'] = avg_rating
	return df.sort_values('counts', ascending=False)


def popular_matrix(ratings_df, top_N):
	users = set(ratings_df['userId'])
	movies = set(ratings_df['movieId'])
	popular_df = get_popular_movies_df(ratings_df).head(top_N)
	rows = popular_df.to_records()
	user_ind = [u for u in users for _ in range(top_N)]
	movie_ind = [m for _ in users for m,_,_ in rows]
	ratings = [r for _ in users for _,_,r in rows]
	um_dense = csr_matrix((ratings, (user_ind, movie_ind)), shape=(len(users), len(movies))).toarray()
	return um_dense



if __name__ == '__main__':
	ratings_df = pd.read_csv('./data/ratings_med.csv')
	print popular_matrix(ratings_df, 10)