import torch
import numpy as np
import random
import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import jit
from functools import partial

@partial(jit,static_argnum=(4,5,))
def gen_random_walk_tensor(walk, graph, node, length, num_walks, rng_key):
    walk = jnp.zeros((num_walks, length), dtype=jnp.int32)
    walk = jax.ops.index_update(walk, jax.ops.index[:, 0], node)
    j = 0
    while j < num_walks:
        current_node = node
        step = 1
        while step < length:
            neighbors = list(graph.neighbors(current_node))
            rng_key, subkey = jrandom.split(rng_key)
            current_node = jrandom.choice(subkey, neighbors)
            walk = jax.ops.index_update(walk, jax.ops.index[j, step], current_node)
            step += 1
        j += 1
    return walk, rng_key

@partial(jit,static_argnum=(2,3,4,))
def gen_batch_random_walk(batch_walk, graph, initial_nodes, length, num_walks, rng_key):
    n_nodes = initial_nodes.shape[0]
    walk = torch.zeros((num_walks, length), dtype=int)
    batch_walk = jnp.zeros((num_walks * n_nodes, length), dtype=jnp.int32)
    for i, n in enumerate(initial_nodes):
        n = n.item()
        rng_key, subkey = jrandom.split(rng_key)
        sub_walk = gen_random_walk_tensor(walk, graph, n, length, num_walks, subkey)
        batch_walk = jax.ops.index_update(batch_walk, jax.ops.index[num_walks * i:num_walks * (i + 1)], sub_walk)
    return walk, batch_walk , rng_key


@jit
def generate_windows(windows, random_walk, window_size):
    num_walks, walk_length = random_walk.shape
    num_windows = walk_length + 1 - window_size
    windows = jnp.zeros((num_walks * num_windows, window_size), dtype=jnp.int32)
    for j in range(num_windows):
        windows = jax.ops.index_update(windows, jax.ops.index[num_walks * j:num_walks * (j + 1)], random_walk[:, j:j + window_size])
    return windows


@jit
def get_windows_dotproduct(windows, embedding):
    embedding_size = embedding.shape[1]
    num_windows, window_size = windows.shape

    # get the embedding of the initial node repeated num_windows times
    first_emb = embedding[windows[:, 0]]
    first_emb = jnp.expand_dims(first_emb, axis=1)  # Add a new axis

    # get the embedding of the remaining nodes in each window
    others_emb = embedding[windows[:, 1:]]
    others_emb = others_emb.reshape(num_windows, window_size - 1, embedding_size)
    # result has same shape as others
    # Each element is the dot product between the corresponding node embedding
    # and the embedding of the first node of that walk
    # that is, result_{i, j} for random walk i and element j is v_{W_{i, 0}} dot v_{W_{i, j}}
    # Compute dot product
    result = jnp.sum(first_emb * others_emb, axis=-1)

    return result

@jit
def gen_negative_samples(negative_samples, amount, length, initial_node, number_of_nodes, rng_key):
    negative_samples = jnp.zeros((amount, length), dtype=jnp.int32)
    negative_samples = jax.ops.index_update(negative_samples, jax.ops.index[:, 0], initial_node)
    rng_key, subkey = jrandom.split(rng_key)
    negative_samples[:, 1:] = jrandom.randint(subkey, (amount, length-1), 0, number_of_nodes)
    return negative_samples, rng_key


@partial(jit, static_argnums=(6,))
def gen_batch_negative_samples(negative_samples, amount, length, initial_nodes, number_of_nodes, rng_key, distribution=None):
    num_initial_nodes = initial_nodes.shape[0]
    M = amount * num_initial_nodes
    N = length - 1

    negative_samples = jnp.zeros((M, length), dtype=jnp.int32)
    negative_samples = jax.ops.index_update(negative_samples, jax.ops.index[:, 0], jnp.repeat(initial_nodes, amount))

    if distribution is None:
        rng_key, subkey = jrandom.split(rng_key)
        negative_samples[:, 1:] = jrandom.randint(subkey, (M, N), 0, number_of_nodes)
    else:
        rng_key, subkey = jrandom.split(rng_key)
        samples = jrandom.choice(subkey, jnp.arange(number_of_nodes), shape=(M * N,), p=distribution)
        negative_samples[:, 1:] = jnp.reshape(samples, (M, N))

    return negative_samples, rng_key
