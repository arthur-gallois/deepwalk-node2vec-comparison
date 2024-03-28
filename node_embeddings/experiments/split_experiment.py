from os.path import join
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from node_embeddings.model import deepWalk, node2vec
from node_embeddings.experiments.base import save_figure, init
from sklearn.model_selection import train_test_split
from node_embeddings.dataset import BlogCatalogDataset, PPIDataset
from sklearn.multioutput import MultiOutputClassifier

import random
import argparse

datasets = {
    'ppi': PPIDataset,
    'blogcatalog': BlogCatalogDataset
}

parser = argparse.ArgumentParser(prog='Split Experiment')
parser.add_argument('-d', '--dataset', choices=datasets.keys(), required=True)   
parser.add_argument('-a', '--algorithm', choices=['node2vec', 'deepwalk'], required=True)  
parser.add_argument('--no_train', action='store_true')  
args = parser.parse_args()

dataset = args.dataset
algorithm = args.algorithm
train = not args.no_train

EXPERIMENT_NAME = f'split_experiment_{dataset}_{algorithm}_longer'
EXPERIMENT_OUTPUT_PATH = join('output', EXPERIMENT_NAME)
SEED=1234


classifiers = {
    'ppi': MultiOutputClassifier(estimator=LogisticRegression(random_state=0)),
    'blogcatalog': LogisticRegression(random_state=0, multi_class='ovr')
}

hyperparameters = {
    'ppi': {'p': 2, 'q': 3},
    'blogcatalog': {'p': 0.25, 'q': 0.25},
}

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
    init(path=EXPERIMENT_OUTPUT_PATH, seed=SEED)

    bc_dataset = datasets[dataset]().load()
    classifier = classifiers[dataset]

    if train:
        if algorithm == 'deepwalk':
            embedding, loss_history = deepWalk(
                graph=bc_dataset['graph'],  
                walks_per_vertex=10, 
                walk_length=40, 
                window_size=10,  
                embedding_size=128,
                num_neg=10,
                lr=1e-2,
                epochs=10,
                batch_size=50
            )
        elif algorithm == 'node2vec':
            embedding, loss_history = node2vec(
                graph=bc_dataset['graph'],  
                walks_per_vertex=10, 
                walk_length=40, 
                window_size=10,  
                embedding_size=128,
                num_neg=5,
                lr=1e-2,
                epochs=10,
                batch_size=50,
                p=hyperparameters[dataset]['p'],
                q=hyperparameters[dataset]['q']
            )
        torch.save(embedding, join(EXPERIMENT_OUTPUT_PATH, 'embedding.pt'))
        with open(join(EXPERIMENT_OUTPUT_PATH, 'loss_history.pickle'), 'wb') as handle:
            pickle.dump(bc_dataset, handle)
        cumsum_vec = np.cumsum(np.insert(loss_history['total'], 0, 0)) 
        window_width = 10
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        plt.plot(ma_vec)
        save_figure(EXPERIMENT_NAME, 'convergence', 'png')
    else:
        embedding = torch.load(join(EXPERIMENT_OUTPUT_PATH, 'embedding.pt'))
    
    labeled_portions = np.linspace(0.1, 0.9, 9)
    for labeled_portion in labeled_portions:
        print('Labeled portion:', labeled_portion, end='')
        microf1, macrof1 = evaluate(embedding, bc_dataset['labels'], classifier = classifier, labeled_portion=labeled_portion)
        print(f'Micro F1: {microf1:.2%} Macro F1: {macrof1:.2%}')
        print()