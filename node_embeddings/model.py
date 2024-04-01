from tqdm import tqdm
import torch
import numpy as np
import random
from .random_walk import generate_windows, gen_batch_random_walk, get_windows_dotproduct, gen_batch_negative_samples,gen_batch_biaised_random_walk

eps = 1e-15

def generate_batches(array, batch_size):
    """Yield successive batches of size `batch_size` from `array`."""
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]

def deepWalk(graph, walks_per_vertex, walk_length, window_size, embedding_size, num_neg, lr, epochs, batch_size, distribution=None):
    number_of_nodes = graph.number_of_nodes()
    
    embedding = (torch.randn(size=(number_of_nodes, embedding_size))).detach()
    embedding.requires_grad = True
    optimizer = torch.optim.Adam([embedding], lr=lr)
    loss_history = {'pos': [], 'neg': [], 'total': []}

    for _ in range(epochs):
        nodes = torch.tensor(list(graph.nodes), dtype=int)
        random_ixs = torch.randperm(nodes.shape[0])
        nodes = nodes[random_ixs]
        node_loader = generate_batches(nodes, batch_size)
        n_batches = int(number_of_nodes / batch_size)
        for n in tqdm(node_loader, total=n_batches):
            random_walk = gen_batch_random_walk(graph, n, walk_length, walks_per_vertex)
            # Positive Sampling
            # each row of windows is one window, we have B = walks_per_vertex*num_windows windows
            windows = generate_windows(random_walk, window_size)
            batch_dotproduct = get_windows_dotproduct(windows, embedding)
            # takes the sigmoid of the dot product to get probability, then
            # takes the loglik and average through all elements
            pos_loss = -torch.log(torch.sigmoid(batch_dotproduct)+eps).mean()
            # Negative Sampling
            negative_samples = gen_batch_negative_samples(
                amount=num_neg*walks_per_vertex, 
                length=walk_length, 
                initial_nodes=n, 
                number_of_nodes=number_of_nodes,
                distribution=distribution
            )
            windows = generate_windows(negative_samples, window_size)
            batch_dotproduct = get_windows_dotproduct(windows, embedding)
            neg_loss = -torch.log(1-torch.sigmoid(batch_dotproduct)+eps).mean()

            loss = pos_loss + neg_loss
            # Optimization
            loss.backward()
            loss_history['total'].append(loss.detach().numpy())
            loss_history['pos'].append(pos_loss.detach().numpy())
            loss_history['neg'].append(neg_loss.detach().numpy())
            optimizer.step()
            optimizer.zero_grad()  

    return embedding, loss_history

def node2vec(graph, walks_per_vertex, walk_length, window_size, embedding_size, num_neg, lr, epochs, batch_size, p = 5, q = 5):
    number_of_nodes = graph.number_of_nodes()
    
    embedding = (torch.randn(size=(number_of_nodes, embedding_size)) ).detach()
    embedding.requires_grad = True
    optimizer = torch.optim.Adam([embedding], lr=lr)
    loss_history = {'pos': [], 'neg': [], 'total': []}
    neighbors_dict = {node: list(graph.neighbors(node)) for node in graph.nodes}

    for _ in range(epochs):
        nodes = torch.tensor(list(graph.nodes), dtype=int)
        random.shuffle(nodes)
        node_loader = generate_batches(nodes, batch_size)
        n_batches = int(number_of_nodes / batch_size)
        for n in tqdm(node_loader, total=n_batches):
            random_walk = gen_batch_biaised_random_walk(graph, n, walk_length, walks_per_vertex, p, q, neighbors_dict)
            num_windows = walk_length + 1 - window_size

            # Positive Sampling
            # each row of windows is one window, we have B = walks_per_vertex*num_windows windows
            windows = generate_windows(random_walk, window_size)
            batch_dotproduct = get_windows_dotproduct(windows, embedding)
            # takes the sigmoid of the dot product to get probability, then
            # takes the loglik and average through all elements
            pos_loss = -torch.log(torch.sigmoid(batch_dotproduct)+eps).mean()
            # Negative Sampling
            negative_samples = gen_batch_negative_samples(
                amount=num_neg*walks_per_vertex, 
                length=walk_length, 
                initial_nodes=n, 
                number_of_nodes=number_of_nodes
            )
            windows = generate_windows(negative_samples, window_size)
            batch_dotproduct = get_windows_dotproduct(windows, embedding)
            neg_loss = -torch.log(1-torch.sigmoid(batch_dotproduct)+eps).mean()

            loss = pos_loss + neg_loss
            # Optimization
            loss.backward()
            loss_history['total'].append(loss.detach().numpy())
            loss_history['pos'].append(pos_loss.detach().numpy())
            loss_history['neg'].append(neg_loss.detach().numpy())
            optimizer.step()
            optimizer.zero_grad()
    return embedding, loss_history