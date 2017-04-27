from __future__ import division
import pandas as pd
import numpy as np

# TODO: Ensure that groups retain order
def precision_recall_at_N(results_df, test_df, top_N=6):
    results_df = results_df.groupby('user').apply(lambda x: x.head(top_N))
    test_df = test_df.groupby('user').apply(lambda x: x.head(top_N))
    overlap_df = results_df.merge(test_df, how='inner', on=['user', 'movie'])

    # TODO: save lengths instead to save memory
    precision = len(overlap_df) / len(results_df)
    recall = len(overlap_df) / len(test_df)
    return precision, recall


def precision_recall_threshold(results_df, test_df, thresh=3.0):
    results_df = results_df[results_df['predicted_rating'] >= thresh]
    test_df = test_df[test_df['actual_rating'] >= thresh]
    overlap_df = results_df.merge(test_df, how='inner', on=['user', 'movie'])

    # TODO: save lengths instead to save memory
    precision = len(overlap_df) / len(results_df)
    recall = len(overlap_df) / len(test_df)
    return precision, recall


def ndcg(results_df, test_df, thresh=3.0):
    def rank(df):
        df['rank'] = range(1,len(df)+1)

    results_df = results_df[results_df['predicted_rating'] >= thresh]
    results_df = results_df.groupby('user').apply(rank)

    test_df = test_df[test_df['actual_rating'] >= thresh]
    test_df = test_df.groupby('user').apply(rank)
    overlap_df = results_df.merge(test_df, how='inner', on=['user', 'movie'], suffixes=('_predicted', '_actual'))

    dcg = overlap_df.groupby('user').apply(lambda x: np.sum(x['predicted_rating'] / np.log2(x['rank_predicted'] + 1)))
    idcg = overlap_df.groupby('user').apply(lambda x: np.sum(x['actual_rating'] / np.log2(x['rank_actual'] + 1)))
    return np.mean(dcg / idcg)


