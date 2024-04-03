from os.path import join
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

from node_embeddings.model import deepWalk
from node_embeddings.dataset import get_dataset
from node_embeddings.experiments.base import save_figure, init
from node_embeddings.random_walk import gen_batch_random_walk

EXPERIMENT_NAME = 'compare_distributions'
EXPERIMENT_OUTPUT_PATH = join('output', EXPERIMENT_NAME)

def generate_unigram_distribution(graph):
    print('Generating distributions - this might take a while...')
    rw = gen_batch_random_walk(
        graph, 
        initial_nodes=torch.tensor(graph.nodes, dtype=int), 
        length=100, 
        num_walks=100
    )
    _, counts = np.unique(rw, return_counts=True)
    unigram_distribution = torch.tensor(counts) / counts.sum()
    return unigram_distribution

def get_scores(evaluator, labels, labeled_portion, embedding):
    NUM_RUNS = 10
    micro_f1_score = 0
    macro_f1_score = 0
    for _ in range(NUM_RUNS):
        _, microf1, macrof1 = evaluator.evaluate(embedding, 
                                    labels, 
                                    labeled_portion=labeled_portion)
        micro_f1_score += microf1
        macro_f1_score += macrof1
    print(f'Micro F1: {micro_f1_score/NUM_RUNS:.2%}, Macro F1: {macro_f1_score/NUM_RUNS:.2%}')
    print()
    return micro_f1_score/NUM_RUNS, macro_f1_score/NUM_RUNS

if __name__ == '__main__':
    init(path=EXPERIMENT_OUTPUT_PATH, seed=1234)
    dataset = get_dataset('ppi')
    data = dataset.load()

    unigram_distribution = generate_unigram_distribution(data['graph'])
    unigram_distribution_power = unigram_distribution**(3/4)
    unigram_distribution_power = unigram_distribution_power/unigram_distribution_power.sum()

    distributions = {
        'unigram': unigram_distribution,
        'powered_unigram': unigram_distribution_power,
        'uniform': None
    }

    for distribution_name, distribution in distributions.items():
        print('Distribution: ', distribution_name)
        embedding, loss_history = deepWalk(
            graph=data['graph'],  
            walks_per_vertex=40, 
            walk_length=80, 
            window_size=10,  
            embedding_size=128,
            num_neg=2,
            lr=1e-1,
            epochs=1,
            batch_size=64,
            distribution=distribution
        )
        torch.save(embedding, join(EXPERIMENT_OUTPUT_PATH, f'embedding_{distribution_name}.pt'))
        with open(join(EXPERIMENT_OUTPUT_PATH, 'loss_history.pickle'), 'wb') as handle:
            pickle.dump(loss_history, handle)

        cumsum_vec = np.cumsum(np.insert(loss_history['total'], 0, 0)) 
        window_width = 10
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        plt.plot(ma_vec)
        save_figure(EXPERIMENT_NAME, f'convergence_{distribution_name}', 'png')

        get_scores(dataset.get_evaluator(), data['labels'], labeled_portion=0.5, embedding=embedding)