from __future__ import division
import pandas as pd
import numpy as np


class BipartiteGraphNode(object):
	def __init__(self, node_id, node_type):
		super(BipartiteGraphNode, self).__init__()
		self.node_id = node_id
		self.node_type = node_type
		self.cpt = dict()
		self.cume_pt = dict()

	def add_edge(self, target_node, edge_weight):
		self.cpt[target_node] = edge_weight

	def normalize_cpt(self):
		rating_sum = sum(self.cpt.values())
		self.cpt = {node : prob / rating_sum for node, prob in self.cpt.items()}

	def cumulative_distribution(self):
		min_prob = 0
		for node, prob in self.cpt.items():
			self.cume_pt[node] = (min_prob, min_prob+self.cpt[node])
			min_prob += self.cpt[node]


def ratings_to_graph(rating_df):
	users = set(rating_df['userId'])
	movies = set(rating_df['movieId'])
	user_nodes = {user: BipartiteGraphNode(user, 'user') for user in users}
	movie_nodes = {movie: BipartiteGraphNode(movie, 'movie') for movie in movies}

	print 'Adding Edges'
	for _, user, movie, rating, _ in rating_df.to_records():
		user_nodes[user].add_edge(movie_nodes[movie], rating)
		movie_nodes[movie].add_edge(user_nodes[user], rating)

	print 'Normalizing Users'
	for user_node in user_nodes.values():
		user_node.normalize_cpt()
		user_node.cumulative_distribution()

	print 'Normalizing Movies'
	for movie_node in movie_nodes.values():
		movie_node.normalize_cpt()
		movie_node.cumulative_distribution()

	return user_nodes, movie_nodes


def assign_user_particles(user_nodes, particles_per_node=20):
	particles = [user_nodes[p % len(user_nodes)] for p in range(particles_per_node * len(user_nodes))]
	return particles


def assign_movie_particles(movie_nodes, particles_per_node=30):
	particles = [movie_nodes[p % len(movie_nodes)] for p in range(particles_per_node * len(movie_nodes))]
	return particles


def update_particle(particle):
	sample = np.random.rand()
	new_particle = [node for node, bounds in particle.cume_pt.items() if bounds[0] <= sample <= bounds[1]][0]
	return new_particle


def filtering(particles, num_iterations=10):
	for i in range(num_iterations):
		# print i
		particles = [update_particle(particle) for particle in particles]
		particles = [update_particle(particle) for particle in particles]
	return particles


def distribution(particles):
	counts = dict()
	for particle in particles: counts[particle] = counts.get(particle, 0) + 1
	dist = pd.Series({particle.node_id: (count / len(particles)) for particle, count in counts.items()})
	return dist

def user_particle_filter():
    rating_df = pd.read_csv('./data/ratings_med.csv')
    user_nodes, _ = ratings_to_graph(rating_df)

    user_particles = assign_user_particles(user_nodes)
    user_particles = filtering(user_particles)
    user_distribution = distribution(user_particles)
    return user_distribution

def movie_particle_filter():
    rating_df = pd.read_csv('./data/ratings_med.csv')
    _, movie_nodes = ratings_to_graph(rating_df)

    movie_particles = assign_movie_particles(movie_nodes)
    movie_particles = filtering(movie_particles)
    movie_distribution = distribution(movie_particles)
    return movie_distribution


if __name__ == '__main__':
	user_particle_filter()
	movie_particle_filter()
