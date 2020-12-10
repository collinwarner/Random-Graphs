import matplotlib.pyplot as plt
from math import comb
from numpy.random import default_rng, SeedSequence, uniform, Generator
import multiprocessing
import concurrent.futures
import numpy as np


def get_edge(w, p):
    if w>p:
        return 1
    return 0


def generate_no_stars_part1(n, p):
    edges = []
    for i in range(comb(n, 2)):
       edges.append(np.random.uniform())
    graph = np.zeros((n, n))
    seen_before = set()
    for i in range(len(graph)):
       for j in range(len(graph[0, :])-1):
           if i != j and (i, j) not in seen_before:
               seen_before.add((i, j))
               seen_before.add((j, i))
               edge = edges.pop()
               graph[i, j] = edge
               graph[j, i] = edge
    return graph

    
def generate_no_stars_part2(graph, p):
    for i in range(len(graph)):
        print(f"edge {i}")
        for j in range(i, len(graph)):
            for k in range(j, len(graph)):
                for l in range(j, len(graph)):
                    for m in range(j, len(graph)):
                        edge1 = graph[i, j]
                        edge2 = graph[j, k]
                        edge3 = graph[i, l]
                        edge4 = graph[i, m]
                        while edge1 + edge2 + edge3 + edge4 < 4*p:
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


def generate_no_triangles_part1(n, p):
    edges = []
    for i in range(comb(n, 2)):
       edges.append(np.random.uniform())
    graph = np.zeros((n, n))
    seen_before = set()
    for i in range(len(graph)):
       for j in range(len(graph[0, :])-1):
           if i != j and (i, j) not in seen_before:
               seen_before.add((i, j))
               seen_before.add((j, i))
               edge = edges.pop()
               graph[i, j] = edge
               graph[j, i] = edge
    return graph






    
def generate_no_triangles_part2(graph, p):
    for i in range(len(graph)):
        # print(f"edge {i}")
        for j in range(i, len(graph)):
            for k in range(j, len(graph)):
                edge1 = graph[i, j]
                edge2 = graph[j, k]
                edge3 = graph[k, i]
                while edge1 + edge2 + edge3 < 3*p:
                    edge1 = np.random.uniform()
                    edge2 = np.random.uniform()
                    edge3 = np.random.uniform()
                graph[i, j] = edge1
                graph[j, k] = edge2
                graph[k, i] = edge3
                graph[j, i] = edge1
                graph[k, j] = edge2
                graph[i, k] = edge3 

    return graph
            


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

def dfs(adj, s, parent = None, order = None):
    if parent is None:
        parent = [None for v in adj]
        order = []

    for v in adj[s]:
        if parent[v] is None:
            parent[v] = s
            dfs(adj, v, parent, order)
    order.append(s)
    return parent, order 

def is_connected(graph, p):  
    adj = [None ]*graph.shape[0]

    for i in range(graph.shape[0]):
        neighbors = []
        for j in range(graph.shape[0]):
            if i != j and graph[i, j] > p:
                neighbors.append(j)
        adj[i] = neighbors
    
    parent = bfs(adj, 0)

    for v in parent:
        if v is None:
            return False
    return True

    
def get_sizes(graph, p):
    adj = [None ]*graph.shape[0]
    for i in range(graph.shape[0]):
        neighbors = []
        for j in range(graph.shape[0]):
            if i != j and graph[i, j] <= p:
                neighbors.append(j)
        adj[i] = neighbors
    
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

def run_sim(graph, n, num_trials, num_p):
    s_p = np.log(n)/(2*n)
    p_range = []
    diff = np.log(n)/n
    p_epsilon = diff/num_p 
    s_p += p_epsilon
    while s_p <=3*np.log(n)/(2*n):
        p_range.append(s_p)
        s_p += p_epsilon
    trials = []    
    for i, p in enumerate(p_range):
        current_trial = []
        for _ in range(num_trials):
            print("creating graph")
            adj = graph(n, p)
            print("finished graph")
            current_trial.append(is_connected(adj, p))
            sizes = get_sizes(adj, p)

            print(f"finished trial | num isolated {count_isolated(sizes)}")
        trials.append((p, current_trial, count_isolated(sizes)/n))
        print(f"Iteration {i} || finished p {p}")
    return trials



def triangle_free(n, p):
    return generate_no_triangles_part2(generate_no_triangles_part1(n, p), p)

def plot_random_graph(model1, model2, n, remove_features = lambda x : x):
    fig, axs = plt.subplots(2)

    num_trials = 1
    num_p = 100
    trials1 = run_sim(model1, n, num_trials, num_p)
    trials2 = run_sim(model2, n, num_trials, num_p)
    x1 = []
    y1 = []
    xi1 = []
    yi1 = []
    for val in trials1:
        p, connected_vals, num_isolated = val
        for c in connected_vals:
            x1.append(p)
            y1.append(1 if c else 0)
        xi1.append(p)
        yi1.append(num_isolated)
    x2 = []
    y2 = []
    xi2 = []
    yi2 = []
    for val in trials2:
        p, connected_vals, num_isolated = val
        for c in connected_vals:
            x2.append(p)
            y2.append(1 if c else 0)
        xi2.append(p)
        yi2.append(num_isolated)
    #plt.clear()
    axs[0].plot(x1, y1, 'o', color='blue')
    axs[0].plot(x2, y2, 'o', color='orange')
    threshold = np.log(n)/n
    threshold_x = [threshold, threshold]
    threshold_y = [0, 1]
    axs[0].plot(threshold_x, threshold_y, color='red')
    axs[1].plot(xi1, yi1, color='blue')
    axs[1].plot(xi2, yi2, color='orange')

    plt.show()
    

    
def run_sim_giant_comp(graph, n, num_trials, num_p):
    lam = 0.5
    s_p = lam/n
    p_range = []
    diff = 1
    l_epsilon = diff/num_p
    while lam <=1.5:
        p_range.append(s_p)
        lam += l_epsilon
        s_p = lam/n
    # p_range = [p/100 for p in range(1, 100)]
    trials = []    
    for i, p in enumerate(p_range):
        current_trial = []
        for _ in range(num_trials):
            print("creating graph")
            adj = graph(n, p)
            print("finished graph")
            sizes = get_sizes(adj, p)
            sizes.sort()
            biggest = sizes[-1]
            second_biggest = 0 if len(sizes) <2 else sizes[-2]
            current_trial.append((biggest/n, second_biggest/biggest))
            print(f"finished trial | sizes sum to {sum(sizes)}, largest size {biggest}, second largest {second_biggest}, num isolated {count_isolated(sizes)}")
        trials.append((p, current_trial))
        print(f"Iteration {i} || finished p {p}")
    return trials




def count_isolated(sizes):
    num_i = 0
    for size in sizes:
        if size == 1:
            num_i +=1
    return num_i
        
def plot_random_graph_giant_comp(model, n, remove_features = lambda x : x):
    num_trials = 1
    num_p = 100
    trials = run_sim_giant_comp(lambda np, pp : model(np, pp), n, num_trials, num_p)
    x = []
    y = [] #biggest vs n
    z = [] #biggest vs second biggest
    for val in trials:
        p, size_vals = val
        for c in size_vals:
            x.append(p)
            y.append(c[0])
            z.append(c[1])
    
    plt.plot(x, y, 'o', color='black')
    threshold = np.log(n)/n
    threshold_x = [threshold, threshold]
    threshold_y = [0, 1]
    # plt.plot(threshold_x, threshold_y, color='red')
    plt.show()