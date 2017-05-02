import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()
from metrics import *
from particle_filtering import *
from similarity import *
import os

# Plot Precision-Recall vs N
def pr_curve_at_N(recs_df, test_df):
	n_arr = range(10, 200, 10)
	precisions, recalls = zip(*[precision_recall_at_N(recs_df, test_df, top_N=n)
		for n in n_arr])

	plt.plot(recalls, precisions)
        plt.grid(True)
        plt.title("Precision vs. Recall at N")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        # The y-axis label is cutoff unless we do this
        plt.tight_layout()

        plt.savefig('./plots/pr_curve_at_N.png')


# Plot Precision-Recall vs Threshold
def pr_curve_thresh(recs_df, test_df):
	thresholds = np.arange(.5, 4.6, .5)
	precisions, recalls = zip(*[precision_recall_threshold(recs_df, test_df, thresh=t)
		for t in thresholds])

	plt.plot(recalls, precisions)
        plt.grid(True)
        plt.title("Precision vs. Recall at threshold")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        # The y-axis label is cutoff unless we do this
        plt.tight_layout()

        plt.savefig('./plots/pr_curve_thresh.png')

# Plot Metrics vs. Alpha


# Plot ROC Curve
def roc_curve(recs_df, test_df):
	thresholds = np.arange(1.01, 5.1, .33)
	fpr, tpr = zip(*[fpr_tpr_threshold(recs_df, test_df, thresh=t)
		for t in thresholds])

	plt.plot(fpr, tpr)
        plt.grid(True)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        # The y-axis label is cutoff unless we do this
        plt.tight_layout()

        plt.savefig('./plots/roc_thresh.png')


# Plot Particle Filtering Convergence
def plot_pf_convergence(ratings_df):
	user_nodes, _ = ratings_to_graph(rating_df)
	user_particles = assign_user_particles(user_nodes)
	iterations = range(10)
	distances = []

	for i in iterations:
		print i
		old_particles = user_particles
		user_particles = filtering_iteration(user_particles)
		distances += [distance(old_particles, user_particles)]

	plt.plot(iterations, distances, 'b-')
	#plt.show()
	fig = plt.figure()
	fig.savefig('./plots/particle_filtering_convergence.png')

if __name__ == '__main__':
	if not os.path.isdir("./plots"):
            os.mkdir("plots")

        recs_df = pd.read_csv('./recs.csv')
	test_df = pd.read_csv('./test.csv')
	ratings_df = pd.read_csv('./data/ratings_med.csv')
#	plot_pf_convergence(recs_df, test_df)
        #pr_curve_at_N(recs_df, test_df)
        pr_curve_thresh(recs_df, test_df)
        #roc_curve(recs_df, test_df)
