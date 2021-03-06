from __future__ import division
import pandas as pd
import numpy as np

def precision_recall_at_N(recs_df, test_df, top_N=50):
    recs_df = recs_df.groupby('user').apply(lambda x: x.head(top_N))
    test_df = test_df[test_df['actual_rating'] >= 4.0]
    #test_df = test_df.groupby('user').apply(lambda x: x.head(20))
    overlap_df = recs_df.merge(test_df, how='inner', on=['user', 'movie'])

    precision = len(overlap_df) / len(recs_df)
    recall = len(overlap_df) / len(test_df)
    return precision, recall


def confusion_matrix_top_N(recs_df, test_df, top_N=50):
    pos_recs = recs_df.groupby('user').apply(lambda x: x.head(top_N))
    pos_test = test_df[test_df['actual_rating'] >= 4.0]
    #pos_test = recs_df.groupby('user').apply(lambda x: x.head(20))
    p = len(pos_test)
    tp = len(pos_recs.merge(pos_test, how='inner', on=['user', 'movie']))
    fn = p - tp

    neg_recs = recs_df.groupby('user').apply(lambda x: x.tail(len(x) - top_N))
    neg_test = test_df[test_df['actual_rating'] < 4.0]
    #neg_test = recs_df.groupby('user').apply(lambda x: x.tail(len(x) - 20))
    n = len(neg_test)
    tn = len(neg_recs.merge(neg_test, how='inner', on=['user', 'movie']))
    fp = n - tn

    return tp, fn, tn, fp


def classification_report_top_N(recs_df, test_df, top_N=50):
    tp, fn, tn, fp = confusion_matrix_top_N(recs_df, test_df, top_N=top_N)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + fn) / (tp + fn + tn + fp)
    return precision, recall, accuracy


def precision_recall_threshold(recs_df, test_df, thresh=3.5):
    recs_df = recs_df[recs_df['predicted_rating'] >= thresh]
    test_df = test_df[test_df['actual_rating'] >= 4.0]
    overlap_df = recs_df.merge(test_df, how='inner', on=['user', 'movie'])

    precision = len(overlap_df) / len(recs_df)
    recall = len(overlap_df) / len(test_df)
    return precision, recall


def fpr_tpr_threshold(recs_df, test_df, thresh=3.5):
    pos_recs = recs_df[recs_df['predicted_rating'] >= thresh]
    pos_test = test_df[test_df['actual_rating'] >= 4.0]
    tp = len(pos_recs.merge(pos_test, how='inner', on=['user', 'movie']))
    tpr = tp / len(pos_test)

    neg_recs = recs_df[recs_df['predicted_rating'] < thresh]
    neg_test = test_df[test_df['actual_rating'] < 4.0]
    tn = len(neg_recs.merge(neg_test, how='inner', on=['user', 'movie']))
    specificity = tn / len(neg_test)
    fpr = 1.0 - specificity

    return fpr, tpr


def auc_threshold(recs_df, test_df, thresh=3.5):
    pos_recs = recs_df[recs_df['predicted_rating'] >= thresh]
    pos_test = test_df[test_df['actual_rating'] >= 4.0]
    tp = len(pos_recs.merge(pos_test, how='inner', on=['user', 'movie']))

    neg_recs = recs_df[recs_df['predicted_rating'] < thresh]
    neg_test = test_df[test_df['actual_rating'] < 4.0]
    tn = len(neg_recs.merge(neg_test, how='inner', on=['user', 'movie']))

    accuracy = (tp + tn) / len(test_df)
    return accuracy


def confusion_matrix_thresh(recs_df, test_df, thresh=3.5):
    pos_recs = recs_df[recs_df['predicted_rating'] >= thresh]
    pos_test = test_df[test_df['actual_rating'] >= 4.0]
    p = len(pos_test)
    tp = len(pos_recs.merge(pos_test, how='inner', on=['user', 'movie']))
    fn = p - tp

    neg_recs = recs_df[recs_df['predicted_rating'] < thresh]
    neg_test = test_df[test_df['actual_rating'] < 4.0]
    n = len(neg_test)
    tn = len(neg_recs.merge(neg_test, how='inner', on=['user', 'movie']))
    fp = n - tn

    return tp, fn, tn, fp


def classification_report_thresh(recs_df, test_df, thresh=3.5):
    tp, fn, tn, fp = confusion_matrix_thresh(recs_df, test_df, thresh=thresh)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + fn) / (tp + fn + tn + fp)
    return precision, recall, accuracy


def compute_ndcg(recs_df, test_df, thresh=3.5):
    def rank(df):
        df['rank'] = range(1,len(df)+1)
        return df

    recs_df = recs_df[recs_df['predicted_rating'] >= thresh]
    recs_df = recs_df.groupby('user').apply(rank)

    test_df = test_df[test_df['actual_rating'] >= 4.0]
    test_df = test_df.groupby('user').apply(rank)
    overlap_df = recs_df.merge(test_df, how='inner', on=['user', 'movie'], suffixes=('_predicted', '_actual'))

    dcg = overlap_df.groupby('user').apply(lambda x: np.sum(x['actual_rating'] / np.log2(x['rank_predicted'] + 1)))
    idcg = overlap_df.groupby('user').apply(lambda x: np.sum(x['actual_rating'] / np.log2(x['rank_actual'] + 1)))
    ndcg = np.mean(dcg / idcg)
    return ndcg


def compute_rmse(recs_df, test_df):
    overlap_df = recs_df.merge(test_df, how='inner', on=['user', 'movie'])
    rmse = np.sqrt(np.mean((overlap_df['predicted_rating'] - overlap_df['actual_rating']) ** 2))
    return rmse


if __name__ == '__main__':
    # recs_df = pd.read_csv('./recs.csv')
    test_df = pd.read_csv('./test.csv')

    # print classification_report_top_N(recs_df, test_df)

    recs_df = pd.read_csv("recs_movie_centrality_particle_filtering_0.6.csv")

    print("t,tp,fn,tn,fp")
    print(2.5,classification_report_thresh(recs_df, test_df, thresh=2.5))
    print(3.0,classification_report_thresh(recs_df, test_df, thresh=3.0))
    print(3.5,classification_report_thresh(recs_df, test_df, thresh=3.5))
    print(4.0,classification_report_thresh(recs_df, test_df, thresh=4.0))
