import torch 
import numpy as np
import random
from tqdm import tqdm

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