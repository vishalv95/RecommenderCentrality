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
def pr_curve_top_N(recs_df, test_df):
	n_arr = range(10, 500, 25)
	#n_arr = range(10, 200, 10)
	precisions, recalls, _ = zip(*[classification_report_top_N(recs_df, test_df, top_N=n)
		for n in n_arr])

        plt.clf()

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
	thresholds = np.arange(1.0, 4.6, .33)
	precisions, recalls, _ = zip(*[classification_report_thresh(recs_df, test_df, thresh=t)
		for t in thresholds])

        plt.clf()

	plt.plot(recalls, precisions)
        plt.grid(True)
        plt.title("Precision vs. Recall at Threshold")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        # The y-axis label is cutoff unless we do this
        plt.tight_layout()

        plt.savefig('./plots/pr_curve_thresh.png')

def pr_curve_thresh_multiple(recs_dfs, test_df, title, settings):
	thresholds = np.arange(1.0, 4.6, .33)

        curve_arr = []

        for recs_df in recs_dfs:
        	precisions, recalls, _ = zip(*[classification_report_thresh(recs_df, test_df, thresh=t)
		for t in thresholds])
                curve_arr.append((precisions, recalls))

        plt.clf()

        plt_curves = []
        for i,curve in enumerate(curve_arr):
            plt_curve, = plt.plot(curve[1], curve[0], label=settings[i])
            plt_curves.append(plt_curve)

        plt.grid(True)
        plt.title(title)
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plt.legend(plt_curves, settings)

        # The y-axis label is cutoff unless we do this
        plt.tight_layout()

        plt.savefig('./plots/pr_curve_thresh_{}.png'.format(title.replace(" ", "")))

def roc_curve_thresh_multiple(recs_dfs, test_df, title, settings):
	thresholds = np.arange(1.01, 5.1, .33)

        curve_arr = []

        for recs_df in recs_dfs:
           	fpr, tpr = zip(*[fpr_tpr_threshold(recs_df, test_df, thresh=t)
    		    for t in thresholds])

                curve_arr.append((fpr, tpr))

        plt.clf()

        plt_curves = []
        for i,curve in enumerate(curve_arr):
            plt_curve, = plt.plot(curve[0], curve[1], label=settings[i])
            plt_curves.append(plt_curve)

        plt.grid(True)
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.legend(plt_curves, settings)

        # The y-axis label is cutoff unless we do this
        plt.tight_layout()

        plt.savefig('./plots/roc_curve_thresh_{}.png'.format(title.replace(" ", "")))


# Plot Metrics vs. Alpha
def metrics_vs_alpha(grid_search_results, hyperparameters, plot_metric, y_axis_label, title):
	for column, setting in hyperparameters.items(): grid_search_results = grid_search_results[grid_search_results[column] == setting]
	alpha, metric = grid_search_results['alpha'], grid_search_results[plot_metric]

        plt.clf()

	plt.plot(alpha, metric, 'b-')

        plt.grid(True)
        plt.xlabel("$\\alpha$")
        plt.ylabel(y_axis_label)
        plt.title(title)

        plt.tight_layout()

	plt.savefig('./plots/alpha_vs_{}-{}-{}.png'.format(plot_metric, hyperparameters['method'], hyperparameters['centrality_measure']))


# Plot ROC Curve
def roc_curve(recs_df, test_df):
	thresholds = np.arange(1.01, 5.1, .33)
	fpr, tpr = zip(*[fpr_tpr_threshold(recs_df, test_df, thresh=t)
		for t in thresholds])

        plt.clf()

	plt.plot(fpr, tpr)
        plt.grid(True)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        # The y-axis label is cutoff unless we do this
        plt.tight_layout()

        plt.savefig('./plots/roc_thresh.png')


# TODO: Plot Confusion Matrix: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(recs_df, test_df):
	tp, fn, tn, fp = confusion_matrix_threh(recs_df, test_df, thresh=3.0)
	# Rows are actual, columns are predicted
	confusion = np.array([[tp, fn], [tn, fp]])
	return confusion


# Plot Particle Filtering Convergence
def plot_pf_convergence(ratings_df):
	user_nodes, _ = ratings_to_graph(ratings_df)
	user_particles = assign_user_particles(user_nodes)
	iterations = range(10)
	distances = []

	for i in iterations:
		old_particles = user_particles
		user_particles = filtering_iteration(user_particles)
		distances += [distance(old_particles, user_particles)]
		print i, distance(old_particles, user_particles)

	plt.plot(iterations, distances, 'b-')
	#plt.show()
	fig = plt.figure()
	fig.savefig('./plots/particle_filtering_convergence.png')


if __name__ == '__main__':
	if not os.path.isdir("./plots"):
            os.mkdir("plots")

#        recs_df = pd.read_csv('./recs.csv')
#	test_df = pd.read_csv('./test.csv')
#        ratings_df = pd.read_csv('./data/ratings_med.csv')
#        plot_pf_convergence(recs_df, test_df)
#        pr_curve_top_N(recs_df, test_df)
#        pr_curve_thresh(recs_df, test_df)
#        roc_curve(recs_df, test_df)
#        filenames = ["recs_movie_centrality_particle_filtering_0.0.csv",
#                     "recs_movie_centrality_particle_filtering_0.6.csv",
#                     "recs_movie_centrality_lsh_particle_filtering_0.0.csv",
#                     "recs_movie_centrality_lsh_particle_filtering_0.6.csv"]
#
#        recs_dfs = [pd.read_csv(f) for f in filenames]
#        title = "Precision vs. Recall at Threshold"
#        settings = ["Full Item CF", "Augmented Item CF", "kNN Item CF", "Augmented kNN Item CF"]
#        pr_curve_thresh_multiple(recs_dfs, test_df, title, settings)
#        title = "ROC Curves"
#        roc_curve_thresh_multiple(recs_dfs, test_df, title, settings)

        grid_search_pairwise = pd.read_csv("grid_search_results_pairwise.csv")
        grid_search_lsh = pd.read_csv("grid_search_results_lsh.csv")

        hyperparameters = {"method" : "movie_centrality",
                           "centrality_measure" : "particle_filtering"}
        plot_measure = "recall_threshold"
        y_axis = "Recall at Threshold"
        title = "Recall at Threshold vs $\\alpha$ (Pairwise)"
        metrics_vs_alpha(grid_search_pairwise, hyperparameters, plot_measure, y_axis, title)

        hyperparameters = {"method" : "movie_centrality",
                           "centrality_measure" : "particle_filtering"}
        plot_measure = "precision_threshold"
        y_axis = "Precision at Threshold"
        title = "Precision at Threshold vs $\\alpha$ (Pairwise)"
        metrics_vs_alpha(grid_search_pairwise, hyperparameters, plot_measure, y_axis, title)

        hyperparameters = {"method" : "movie_centrality_lsh",
                           "centrality_measure" : "particle_filtering"}
        plot_measure = "precision_threshold"
        y_axis = "Precision at Threshold"
        title = "Precision at Threshold vs $\\alpha$ (LSH)"
        metrics_vs_alpha(grid_search_lsh, hyperparameters, plot_measure, y_axis, title)

        hyperparameters = {"method" : "movie_centrality_lsh",
                           "centrality_measure" : "particle_filtering"}
        plot_measure = "recall_threshold"
        y_axis = "Recall at Threshold"
        title = "Recall at Threshold vs $\\alpha$ (LSH)"
        metrics_vs_alpha(grid_search_lsh, hyperparameters, plot_measure, y_axis, title)

