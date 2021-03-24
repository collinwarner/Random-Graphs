from math import comb

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

"""
Checks if a given edge exists or not

w: number weight of the edge in question
p: probability threshold

Returns: 1 edge exists, 0 otherwise
"""
def get_edge(w, p):
    if w <= p:
        return 1
    return 0


"""
Creates a Erdos Renyi graph
n: number of nodes
p: probability threshold
Returns np array of arrays representing the adjaceny matrix form of the graph (weighted
"""
def generate_erdos_renyi(n, p):
    edges = []
    for i in range(comb(n, 2)):
        edges.append(np.random.uniform())
    graph = np.zeros((n, n))
    seen_before = set()
    for i in range(len(graph)):
        for j in range(len(graph[0, :]) - 1):
            if i != j and (i, j) not in seen_before:
                seen_before.add((i, j))
                seen_before.add((j, i))
                edge = edges.pop()
                graph[i, j] = edge
                graph[j, i] = edge
    return graph

"""
starting point for creating no stars. Right now impractically slow O(n^4)
"""
def generate_no_stars_part2(graph, p):
    for i in range(len(graph)):
        # print(f"edge {i}")
        for j in range(i, len(graph)):
            for k in range(j, len(graph)):
                for l in range(j, len(graph)):
                    for m in range(j, len(graph)):
                        edge1 = graph[i, j]
                        edge2 = graph[j, k]
                        edge3 = graph[i, l]
                        edge4 = graph[i, m]
                        while edge1 + edge2 + edge3 + edge4 < 4 * p:
                            edge1 = np.random.uniform()
                            edge2 = np.random.uniform()
                            edge3 = np.random.uniform()
                            edge4 = np.random.uniform()
                        graph[i, j] = edge1
                        graph[j, k] = edge2
                        graph[k, i] = edge3
                        graph[j, i] = edge1
                        graph[k, j] = edge2
                        graph[i, k] = edge3
    return graph

"""
check if the given indices representing a triangle are in the provided set
i, j, k: numbers of the node in the given triangle
seen_triangles: set of triple tuples representing triangles
returns True if a triangle is in the set (node order doesn't mattter)
"""
def triangle_in_set(i, j, k, triangles_seen):
    return (i, j, k) in triangles_seen or (i, k, j) in triangles_seen or (j, i, k) in triangles_seen or (j, k, i) in triangles_seen or (k, j, i) in triangles_seen or (k, i, j) in triangles_seen

"""
Given an ER graph transforms into triangle free O(n^3)
graph: ER graph in same form that generate_erdos_renyi provides
p: probability threshold
returns: adjaceny matrix graph of the mutated triangle free graph (weighted)
"""
def remove_triangles_from_er_slow(graph, p):
    num_trigs = 0
    changed_edges = set()
    for i in range(len(graph)):
        for j in range(i, len(graph)):
            for k in range(j, len(graph)):
                if i != j and j != k and k != i and not triangle_in_set(i, j, k, changed_edges):
                    changed_edges.add((i, j, k))
                    edge1 = graph[i, j]
                    edge2 = graph[j, k]
                    edge3 = graph[k, i]
                    if edge1 + edge2 + edge3 < 3*p:
                        num_trigs += 1
                    while edge1 + edge2 + edge3 < 3 * p:
                        if edge1 <= p:
                            edge1 = np.random.uniform()
                        elif edge2 <= p:
                            edge2 = np.random.uniform()
                        elif edge3 <= p:
                            edge3 = np.random.uniform()
                        else:
                            print("shouldn't get here")

                    graph[i, j] = edge1
                    graph[j, k] = edge2
                    graph[k, i] = edge3
                    graph[j, i] = edge1
                    graph[k, j] = edge2
                    graph[i, k] = edge3
    print(f"num triangles: {num_trigs}")
    return graph

"""
Given an ER graph transforms into triangle free O(n^2)
graph: ER graph in same form that generate_erdos_renyi provides
p: probability threshold
returns: adjaceny matrix graph of the mutated triangle free graph (weighted)
"""
def remove_triangles_from_er(graph, p):
    edges = set()
    for i in range(len(graph)):
        for j in range(len(graph)):
            if i != j and graph[i, j] <= p and (j, i) not in edges:
                edges.add((i, j))
    num_trigs = 0
    for k in range(len(graph)):
        for i, j in edges:
            if i != j and j != k and k != i:
                edge1 = graph[i, j]
                edge2 = graph[j, k]
                edge3 = graph[k, i]
                if edge1 + edge2 + edge3 <= 3 * p:
                    num_trigs += 1
                    edge1 = np.random.uniform(p, 1)
                graph[i, j] = edge1
                graph[j, i] = edge1
    return graph
"""
BFS
adj: adjacency list representing a graph
s: starting node
Returns: parents of every nodes
"""
def bfs(adj, s):
    parent = [None for v in adj]
    parent[s] = s
    level = [[s]]
    while 0 < len(level[-1]):
        level.append([])
        for u in level[-2]:
            for v in adj[u]:
                if parent[v] is None:
                    parent[v] = u
                    level[-1].append(v)
    return parent

"""
Convert Adjaceny matrix to adjaceny list
graph: Adjaceny matrix form of graph
p: probabiliy threshold
Returns: adjaceny list form with edges realized
"""
def create_adj_graph(graph, p):
    adj = [None] * graph.shape[0]

    for i in range(graph.shape[0]):
        neighbors = []
        for j in range(graph.shape[0]):
            if i != j and graph[i, j] <= p:
                neighbors.append(j)
        adj[i] = neighbors
    return adj

"""
Checks if a graph is connected
"""
def is_connected(graph, p):
    adj = create_adj_graph(graph, p)
    parent = bfs(adj, 0)

    for v in parent:
        if v is None:
            return 0
    return 1

"""
Gets the sizes of the different components of the input graph
"""
def get_sizes(graph, p):
    adj = create_adj_graph(graph, p)
    sizes = []
    parent = [None for v in adj]
    for s in range(len(adj)):
        if parent[s] is None:
            parent[s] = s
            level = [[s]]
            size = 1
            while 0 < len(level[-1]):
                level.append([])
                for u in level[-2]:
                    for v in adj[u]:
                        if parent[v] is None:
                            parent[v] = u
                            level[-1].append(v)
                            size += 1
            sizes.append(size)
    return sizes

"""
Counts the number of isolated vertices
"""
def count_isolated(sizes):
    num_i = 0
    for size in sizes:
        if size == 1:
            num_i += 1
    return num_i

"""
Given an input graph_type (ex triangle_free) and a range of p values, 
runs a simulation on each p value creating graphs of type graph_type and n nodes

The simulation returns the connectivity, size of giant, and number of isolated points for each of the p-values
"""
def run_sim(graph_type, n, p_range):
    sim_data_for_each_p = []
    for i, p in enumerate(p_range):
        print("creating graph")
        adj = graph_type(n, p)
        sizes = get_sizes(adj, p)
        print("finished graph")
        sizes.sort()
        biggest = sizes[-1]
        second_biggest = 0 if len(sizes) < 2 else sizes[-2]
        sim_data_for_each_p.append((is_connected(adj, p), biggest / n, second_biggest / biggest, count_isolated(sizes) / n))
        print(f"finished trial | num isolated {count_isolated(sizes)}")
        print(f"Iteration {i} || finished p {p}")

    connected_status = []
    size_of_giant = []
    isolated_values = []
    for val in sim_data_for_each_p:
        connected, giant, ratio_of_giant_to_next, fraction_isolated = val
        connected_status.append(connected)
        size_of_giant.append(giant)
        isolated_values.append(fraction_isolated)
    print("p range", p_range)
    return connected_status, size_of_giant, isolated_values

"""
Creates a triangle free graph of size n, with probability threshold p
"""
def triangle_free(n, p):
    return remove_triangles_from_er(generate_erdos_renyi(n, p), p)

"""
Creates a star free graph of size n, with probability threshold p
"""
def star_free(n, p):
    return generate_no_stars_part2(generate_erdos_renyi(n, p), p)

"""
Generates p-values in the range [ln(n)/2n, 3ln(n)/2] which is used to 
narrow in on the connectivity threshold
"""
def create_connected_p_range(n, num_p):
    starting_p = np.log(n) / (2 * n)
    p_range = []
    diff = np.log(n) / n
    p_epsilon = diff / num_p
    while starting_p <= 3 * np.log(n) / (2 * n):
        p_range.append(starting_p)
        starting_p += p_epsilon
    return p_range

"""
Generates p-values in the range [1/2n, 3/(2n)] which is used to 
see the growth of the giant component
"""
def create_giant_p_range(n, num_p):
    lam = 0.5
    starting_p = lam/n
    p_range = []
    diff = 1
    l_epsilon = diff/num_p
    while lam <= 1.5:
        p_range.append(starting_p)
        lam += l_epsilon
        starting_p = lam/n
    return p_range

"""
Constructs plots showing the graphs for connectivity vs p and size of giant vs p. 
The p range depends on test_connected, and if test_connected is true the p values will 
narrow in on the threshold, otherwise, narrows on the giant property
"""
def plot_random_graph(model1, model2, n, test_connected=True):
    fig, axs = plt.subplots(3)
    num_p = 100

    p_range = create_connected_p_range(n, num_p) if test_connected else create_giant_p_range(n, num_p)


    connected_status1, size_of_giant1, isolated_values1 = run_sim(model1, n, p_range)
    connected_status2, size_of_giant2, isolated_values2 = run_sim(model2, n, p_range)

    axs[0].set_title("Connectivity vs p")
    axs[0].plot(p_range, connected_status1, 'o', color='blue')
    axs[0].plot(p_range, connected_status2, 'o', color='orange')
    threshold = np.log(n) / n
    threshold_x = [threshold, threshold]
    threshold_y = [0, 1]
    axs[0].plot(threshold_x, threshold_y, color='red')

    axs[1].set_title("Fraction Isolated vs p")
    axs[1].plot(p_range, isolated_values1, color='blue')
    axs[1].plot(p_range, isolated_values2, color='orange')

    axs[2].set_title("Size of Giant vs p")
    axs[2].plot(p_range, size_of_giant1, 'o', color='blue')
    axs[2].plot(p_range, size_of_giant2, 'o', color='orange')

    plt.show()




#README, LATEX explaining graphs, and finish docs
if __name__ == "__main__":
    n = 100
    plot_random_graph(generate_erdos_renyi, triangle_free, n, test_connected=False)
