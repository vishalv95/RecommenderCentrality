from __future__ import division
import pandas as pd
import numpy as np

# TODO: Ensure that groups retain order
def precision_recall_at_N(recs_df, test_df, top_N=6):
    recs_df = recs_df.groupby('user').apply(lambda x: x.head(top_N))
    test_df = test_df.groupby('user').apply(lambda x: x.head(top_N))
    overlap_df = recs_df.merge(test_df, how='inner', on=['user', 'movie'])

    precision = len(overlap_df) / len(recs_df)
    recall = len(overlap_df) / len(test_df)
    return precision, recall


def precision_recall_threshold(recs_df, test_df, thresh=3.0):
    recs_df = recs_df[recs_df['predicted_rating'] >= thresh]
    test_df = test_df[test_df['actual_rating'] >= thresh]
    overlap_df = recs_df.merge(test_df, how='inner', on=['user', 'movie'])

    precision = len(overlap_df) / len(recs_df)
    recall = len(overlap_df) / len(test_df)
    return precision, recall

def auc_threshold(recs_df, test_df, thresh=3.0):
    pos_recs = recs_df[recs_df['predicted_rating'] >= thresh]
    pos_test = test_df[test_df['actual_rating'] >= thresh]
    tp = len(pos_recs.merge(pos_test, how='inner', on=['user', 'movie']))

    neg_recs = recs_df[recs_df['predicted_rating'] < thresh]
    neg_test = test_df[test_df['actual_rating'] < thresh]
    tn = len(neg_recs.merge(neg_test, how='inner', on=['user', 'movie']))

    accuracy = (tp + tn) / len(test_df)
    return accuracy


def compute_ndcg(recs_df, test_df, thresh=3.0):
    def rank(df):
        df['rank'] = range(1,len(df)+1)
        return df

    recs_df = recs_df[recs_df['predicted_rating'] >= thresh]
    recs_df = recs_df.groupby('user').apply(rank)

    test_df = test_df[test_df['actual_rating'] >= thresh]
    test_df = test_df.groupby('user').apply(rank)
    overlap_df = recs_df.merge(test_df, how='inner', on=['user', 'movie'], suffixes=('_predicted', '_actual'))

    dcg = overlap_df.groupby('user').apply(lambda x: np.sum(x['predicted_rating'] / np.log2(x['rank_predicted'] + 1)))
    idcg = overlap_df.groupby('user').apply(lambda x: np.sum(x['actual_rating'] / np.log2(x['rank_actual'] + 1)))
    ndcg = np.mean(dcg / idcg)
    return ndcg


def compute_rmse(recs_df, test_df):
    overlap_df = recs_df.merge(test_df, how='inner', on=['user', 'movie'])
    rmse = np.sqrt(np.mean((overlap_df['predicted_rating'] - overlap_df['actual_rating']) ** 2))
    return rmse


if __name__ == '__main__':
    recs_df = pd.read_csv('./recs.csv')
    test_df = pd.read_csv('./test.csv')

    print auc_threshold(recs_df, test_df, 3.0)