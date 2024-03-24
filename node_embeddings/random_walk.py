import torch
import numpy as np
import random

def gen_random_walk_tensor(graph, node, length, num_walks):
    walk = torch.zeros((num_walks, length), dtype=int)
    walk[:, 0] = node
    j = 0
    while j < num_walks:
        current_node = node
        step = 1
        while step < length:
            neighbors = list(graph.neighbors(current_node))
            current_node = random.choice(neighbors)
            walk[j, step] = current_node
            step += 1
        j+=1
    return walk

def gen_batch_random_walk(graph, initial_nodes, length, num_walks):
    n_nodes = initial_nodes.shape[0]
    walk = torch.zeros((num_walks*n_nodes, length), dtype=int)
    for i, n in enumerate(initial_nodes):
        n = n.item()
        walk[num_walks*i:num_walks*(i+1)] = gen_random_walk_tensor(graph, n, length, num_walks)
    return walk

def generate_windows(random_walk, window_size):
    num_walks, walk_length = random_walk.shape
    # number of windows: e.g. length 5, window size 3 -> 3 windows ([0, 1, 2], [1, 2, 3], [2, 3, 4])
    num_windows = walk_length + 1 - window_size
    windows = torch.zeros((num_walks*num_windows, window_size), dtype=int)
    for j in range(num_windows):
        windows[num_walks*j:num_walks*(j+1)] = random_walk[:, j:j+window_size]
    return windows

def get_windows_dotproduct(windows, embedding):
    embedding_size = embedding.shape[1]
    # get the embedding of the initial node repeated num_windows times
    first_emb = embedding[windows[:, 0]]
    first_emb = first_emb.view(windows.shape[0], 1, embedding_size)
    # get the embedding of the remaining nodes in each window
    others_emb = embedding[windows[:, 1:]]
    others_emb = others_emb.view(windows.shape[0], -1, embedding_size)
    # result has same shape as others
    # Each element is the dot product between the corresponding node embedding
    # and the embedding of the first node of that walk
    # that is, result_{i, j} for random walk i and element j is v_{W_{i, 0}} dot v_{W_{i, j}}
    result = (first_emb*others_emb).sum(dim=-1)
    return result

def gen_negative_samples(amount, length, initial_node, number_of_nodes):
    negative_samples = torch.zeros((amount, length), dtype=int)
    negative_samples[:, 0] = initial_node
    negative_samples[:, 1:] = torch.randint(number_of_nodes, (amount, length-1))
    return negative_samples

def gen_batch_negative_samples(amount, length, initial_nodes, number_of_nodes, distribution=None):
    negative_samples = torch.zeros((amount*initial_nodes.shape[0], length), dtype=int)
    negative_samples[:, 0] = initial_nodes.repeat(amount, 1).t().contiguous().view(-1)
    if distribution is None:
        negative_samples[:, 1:] = torch.randint(number_of_nodes, (amount*initial_nodes.shape[0], length-1))
    else:
        M = amount*initial_nodes.shape[0]
        N = length-1
        negative_samples[:, 1:] = torch.multinomial(distribution, M*N, replacement=True).view(M, N)
    return negative_samples

def gen_biaised_random_walk_tensor(graph, start_node, walk_length, num_walks, p, q , neighbors_dict):
    walks = torch.zeros((num_walks, walk_length), dtype=int)
    walks[:, 0] = start_node

    for walk_index in range(num_walks):
        current_node = start_node
        for step in range(walk_length):
            walks[walk_index, step] = current_node
            if step > 0:
                prev_node = int(walks[walk_index, step - 1])
                current_node = get_next_node(graph,prev_node,current_node,p,q)
            else:
                current_node = np.random.choice(list(graph.neighbors(current_node)))
            
    
    return walks

def get_next_node(graph,t,v, p, q):
    v_neighbors = set(graph.neighbors(v))
    t_neighbors = set(graph.neighbors(t))
    t_set = set([t])

    vt_neighbors = v_neighbors & t_neighbors
    only_v_neighbors = v_neighbors - t_neighbors - t_set

    allsets = [vt_neighbors,only_v_neighbors,t_set]

    vt_weights = 1 * len(vt_neighbors)
    only_v_weights = 1/q * len(only_v_neighbors)
    t_weight = 1/p

    prob_vector = np.array((vt_weights,only_v_weights,t_weight))
    prob_vector = prob_vector / np.sum(prob_vector)

    chosen_set = np.random.choice(allsets,p=prob_vector)
    next_node = np.random.choice(list(chosen_set))
    return next_node

def gen_batch_biaised_random_walk(graph, initial_nodes, length, num_walks, p, q, neighbors_dict):
    n_nodes = initial_nodes.shape[0]
    walk = torch.zeros((num_walks*n_nodes, length), dtype=int)
    for i, n in enumerate(initial_nodes):
        n = n.item()
        walk[num_walks*i:num_walks*(i+1)] = gen_biaised_random_walk_tensor(graph, n, length, num_walks, p, q,neighbors_dict)
    return walk