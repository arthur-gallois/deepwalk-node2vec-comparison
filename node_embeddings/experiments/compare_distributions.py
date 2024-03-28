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
    rw = gen_batch_random_walk(
        graph, 
        initial_nodes=torch.tensor(dataset['graph'].nodes, dtype=int), 
        length=10, 
        num_walks=10
    )
    _, counts = np.unique(rw, return_counts=True)
    unigram_distribution = torch.tensor(counts) / counts.sum()
    return unigram_distribution

def evaluate(embedding, labels, classifier, labeled_portion):
    X = embedding.detach().numpy()
    y = labels
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=(1-labeled_portion))

    clf = classifier.fit(X_train, y_train)
    
    y_hat = clf.predict(X_test)
    
    microf1 = f1_score(y_test, y_hat, average='micro')
    macrof1 = f1_score(y_test, y_hat, average='macro')
    return microf1, macrof1

if __name__ == '__main__':
    init(path=EXPERIMENT_OUTPUT_PATH, seed=1234)
    dataset = get_dataset('ppi').load()

    unigram_distribution = generate_unigram_distribution(dataset['graph'])
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
            graph=dataset['graph'],  
            walks_per_vertex=10, 
            walk_length=40, 
            window_size=10,  
            embedding_size=128,
            num_neg=5,
            lr=1e-2,
            epochs=10,
            batch_size=50,
            distribution=distribution
        )
        torch.save(embedding, join(EXPERIMENT_OUTPUT_PATH, f'embedding_{distribution_name}.pt'))
        with open(join(EXPERIMENT_OUTPUT_PATH, 'loss_history.pickle'), 'wb') as handle:
            pickle.dump(dataset, handle)

        cumsum_vec = np.cumsum(np.insert(loss_history['total'], 0, 0)) 
        window_width = 10
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        plt.plot(ma_vec)
        save_figure(EXPERIMENT_NAME, f'convergence_{distribution_name}', 'png')

        X = embedding.detach().numpy()
        y = dataset['labels']

        clf = MultiOutputClassifier(estimator=LogisticRegression(random_state=0))
        microf1, macrof1 = evaluate(embedding, y, clf, labeled_portion=0.5)

        print('Micro F1: ', microf1)
        print('Macro F1: ', macrof1)