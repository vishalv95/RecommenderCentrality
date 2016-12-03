from __future__ import division 
import numpy as np 
import Queue


def parse_input(filepath):
    return [(int(line.split()[1]), int(line.split()[2]), float(line.split()[3])) 
        for line in open(filepath, 'rb').readlines() if line[0] == 'a']


def parse_input1(filepath):
    return [(int(line.split()[1]) - 1, int(line.split()[2]) - 1, float(line.split()[3])) 
        for line in open(filepath, 'rb').readlines() if line[0] == 'a']


def parse_input2(filepath):
    return [(int(line.split()[1]) - 1, int(line.split()[2]) - 1, 1.0) 
        for line in open(filepath, 'rb').readlines() if line[0] == 'a']


def form_weighted_graph(edge_list):
    adj_list = dict()
    for start, end, weight in edge_list:
        if start not in edge_list:
            adj_list[start] = dict()
        adj_list[start][end] = weight
    return adj_list


def form_graph(edge_list):
    adj_list = dict()
    for start, end, weight in edge_list:
        if start not in edge_list:
            adj_list[start] = set()
        adj_list[start].add(end)
    return adj_list


def form_vertex_set(edge_list):
    # Construct vertex set
    vertex_set = set()
    for start, end, weight in edge_list: 
        vertex_set.add(start)
        vertex_set.add(end)
    return vertex_set


def floyd_warshall_paths(vertex_set, edge_list): 
    # Initialize distance and path matrices 
    V = len(vertex_set)
    dist = np.empty((V, V))
    dist[:] = np.inf
    next_node = [[None]*V for i in range(len(vertex_set))]

    # Initial distances/paths for edges and vertexes
    for d in range(V):
        dist[d][d] = 0 

    for start, end, weight in edge_list:
        dist[start][end] = weight
        next_node[start][end] = end

    # Update path between nodes if intermediate path is shorter
    for k in range(V):
        if k % 1000 == 0: print k
        for i in range(V):
            for j in range(V):
                if dist[i][k] + dist[k][j] <  dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    return next_node


def brandes_bc(graph, vertex_set):
    distance_from_source = dict()
    predecessors = dict()
    num_shortest_paths = dict()
    dependency = dict()
    betweeness_centrality = {w:0 for w in vertex_set}
    for vertex in vertex_set:
        stack = []
        queue = Queue.Queue()
        for w in vertex_set: predecessors[w] = []
        for t in vertex_set: 
            distance_from_source[t] = np.inf
            num_shortest_paths[t] = 0

        distance_from_source[vertex] = 0 
        num_shortest_paths[vertex] = 1

        queue.put(vertex)

        while not queue.empty():
            v = queue.get()
            stack.append(v)

            for w in graph[v]:
                # Path discovery 
                if distance_from_source[w] == np.inf:
                    distance_from_source[w] = distance_from_source[v]  + 1
                    queue.put(w)

                # Path counting
                if distance_from_source[w] == distance_from_source[v] + 1:
                    num_shortest_paths[w] += num_shortest_paths[v]
                    predecessors[w].append(v)

        # Backpropogation 
        for v in vertex_set: dependency[v] = 0
        while stack: 
            w = stack.pop()
            for v in predecessors[w]:
                dependency[v] = dependency[v] + (num_shortest_paths[v] / num_shortest_paths[w]) * (1 + dependency[w])
            if w != vertex:
                betweeness_centrality[w] += dependency[w]
    return betweeness_centrality



# TODO: Make sure vertex_sp_count is updated
def get_shortest_path(next_node, start, end, vertex_sp_count):
    if (next_node[start][end]) is None :
        return

    while start != end:
        start = next_node[start][end]
        vertex_sp_count[start] += 1 


def betweeness_centrality(vertex_set, next_node):
    V = len(vertex_set)
    pair_cardinality = ((V) * (V-1)) 
    vertex_sp_count = {vertex:0 for vertex in vertex_set}
    for start in vertex_set:
        for end in vertex_set:
            get_shortest_path(next_node, start, end, vertex_sp_count)
    return [(vertex + 1, sp_count / pair_cardinality) for vertex, sp_count in vertex_sp_count.items()]


if __name__ == '__main__':
    edge_list = parse_input2('betweeness_tests/USA-road-d.BAY.gr')
    vertex_set = form_vertex_set(edge_list)
    # graph = form_graph(edge_list)
    print "here"
    next_node = floyd_warshall_paths(vertex_set, edge_list)
    # print betweeness_centrality(vertex_set, next_node)
    # print brandes_bc(graph, vertex_set)