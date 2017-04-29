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
		self.cpt = {node : prob / sum(self.cpt.values()) for node, prob in self.cpt.items()}
	
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

	for _, user, movie, rating, _ in rating_df.to_records():
		user_nodes[user].add_edge(movie_nodes[movie], rating)
		movie_nodes[movie].add_edge(user_nodes[user], rating)

	for user_node in user_nodes.values(): 
		user_node.normalize_cpt()
		user_node.cumulative_distribution()

	for movie_node in movie_nodes.values(): 
		movie_node.normalize_cpt()
		movie_node.cumulative_distribution()

	return user_nodes, movie_nodes


def assign_user_particles(user_nodes, num_particles=1e6):
	particles = [user_nodes[p % len(user_nodes)] for p in range(num_particles)]
	return particles


def assign_movie_particles(movie_nodes, num_particles=1e6):
	particles = [movie_nodes[p % len(movie_nodes)] for p in range(num_particles)]
	return particles


def update_particle(particle):
	sample = np.random.rand()
	new_particle = [node for node, bounds in particle.cume_pt if bounds[0] <= sample <= bounds[1]][0]
	return new_particle


def particle_filter(particles, num_iterations=50):
	for i in range(num_iterations):
		print i 
		particles = [update_particle(particle) for particle in particles]
		particles = [update_particle(particle) for particle in particles]
	return particles


def particle_distribution(particles):
	counts = dict()
	for particle in particles: counts[particle] = counts.get(particle, 0) + 1
	dist = {particle: count / len(particles) for particle, count in counts.items()}
	return dist


if __name__ == '__main__':
	rating_df = pd.read_csv('./data/ratings_med.csv')
	user_nodes, movie_nodes = ratings_to_graph(rating_df)

	user_particles = assign_user_particles(user_nodes)
	print particle_distribution(user_particles)

	user_particles = particle_filter(user_particles)
	print particle_distribution(user_particles)

	# movie_particles = assign_movie_particles(movie_nodes)
	# print particle_distribution(movie_particles)

	# movie_particles = particle_filter(movie_particles)
	# print particle_distribution(movie_particles)

