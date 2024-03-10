from tqdm import tqdm
import torch
import numpy as np
import random
import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import jit
from functools import partial
import optax
from jax import grad
from jax_random_walk import generate_windows, gen_batch_random_walk, get_windows_dotproduct, gen_batch_negative_samples 
from node_embeddings.biaised_random_walk import *
eps = 1e-15

@jit
def generate_batches(array, batch_size):
    """Yield successive batches of size `batch_size` from `array`."""
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]


def deepWalk(graph, walks_per_vertex, walk_length, window_size, embedding_size, num_neg, lr, epochs, batch_size, rng_key, distribution=None):
    number_of_nodes = graph.number_of_nodes()
    
    embedding = random.normal(random.PRNGKey(0), (number_of_nodes, embedding_size))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(embedding)
    loss_history = {'pos': [], 'neg': [], 'total': []}
    num_walks, walk_length = random_walk.shape
    num_windows = walk_length + 1 - window_size
    windows = jnp.zeros((num_walks * num_windows, window_size), dtype=jnp.int32)

    @jit
    def loss_fn(params, batch_walk, windows, neg_samples):
        batch_dotproduct = get_windows_dotproduct(windows, params)
        pos_loss = -jnp.mean(jnp.log(jax.scipy.special.expit(batch_dotproduct)))
        
        batch_dotproduct = get_windows_dotproduct(neg_samples, params)
        neg_loss = -jnp.mean(jnp.log(1 - jax.scipy.special.expit(batch_dotproduct)))
        
        loss = pos_loss + neg_loss
        return loss, (pos_loss, neg_loss)

    grad_fn = jit(grad(loss_fn))

    for _ in range(epochs):
        nodes = jnp.array(list(graph.nodes))
        nodes = random.permutation(random.PRNGKey(0), nodes)
        node_loader = generate_batches(nodes, batch_size)
        n_batches = number_of_nodes // batch_size

        for n in tqdm(node_loader, total=n_batches):
            random_walk, batch_walk , rng_key = gen_batch_random_walk(graph, n, walk_length, walks_per_vertex, rng_key)
            windows = generate_windows(windows, random_walk, window_size)
            neg_samples = gen_batch_negative_samples(num_neg * walks_per_vertex, walk_length, n, number_of_nodes, distribution, rng_key )

            params = optimizer.target(opt_state)
            grads, (pos_loss, neg_loss) = grad_fn(params, random_walk, windows, neg_samples)
            updates, opt_state = optimizer.update(grads, opt_state)
            embedding = optax.apply_updates(params, updates)

            loss_history['total'].append(pos_loss + neg_loss)
            loss_history['pos'].append(pos_loss)
            loss_history['neg'].append(neg_loss)

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