from os.path import join
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

from node_embeddings.model import deepWalk, node2vec
from node_embeddings.experiments.base import save_figure, init
from sklearn.model_selection import train_test_split
from node_embeddings.dataset import BlogCatalogDataset, PPIDataset

import random
import argparse

datasets = {
    'ppi': PPIDataset,
    'blogcatalog': BlogCatalogDataset
}

parser = argparse.ArgumentParser(prog='Split Experiment')
parser.add_argument('--dataset', choices=datasets.keys(), required=True)   
parser.add_argument('-a', '--algorithm', choices=['node2vec', 'deepwalk'], required=True)  

parser.add_argument('-t', '--walk_length', type=int, default=80)
parser.add_argument('-w', '--window_size', type=int, default=10)
parser.add_argument('-d', '--embedding_size', type=int, default=128)
parser.add_argument('-k', '--num_negatives', type=int, default=2)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
parser.add_argument('-e', '--epochs', type=int, default=2)
parser.add_argument('-b', '--batch_size', type=int, default=50)
parser.add_argument('-g', '--walks_per_vertex', type=int, default=80)
parser.add_argument('-p', type=float, default=2, help='Parameter p from Node2Vec')
parser.add_argument('-q', type=float, default=3, help='Parameter q from Node2Vec')

parser.add_argument('--no_train', action='store_true')  
args = parser.parse_args()

dataset_name = args.dataset
algorithm = args.algorithm
train = not args.no_train

EXPERIMENT_NAME = f'split_experiment_{dataset_name}_{algorithm}'
EXPERIMENT_OUTPUT_PATH = join('output', EXPERIMENT_NAME)
SEED=1234

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

    dataset = datasets[dataset_name]()
    data = dataset.load()

    if train:
        if algorithm == 'deepwalk':
            embedding, loss_history = deepWalk(
                graph=data['graph'],  
                walks_per_vertex=args.walks_per_vertex, 
                walk_length=args.walk_length, 
                window_size=args.window_size,  
                embedding_size=args.embedding_size,
                num_neg=args.num_negatives,
                lr=args.learning_rate,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
        elif algorithm == 'node2vec':
            embedding, loss_history = node2vec(
                graph=data['graph'],  
                walks_per_vertex=args.walks_per_vertex, 
                walk_length=args.walk_length, 
                window_size=args.window_size,  
                embedding_size=args.embedding_size,
                num_neg=args.num_negatives,
                lr=args.learning_rate,
                epochs=args.epochs,
                batch_size=args.batch_size,
                p=args.p,
                q=args.q
            )
        torch.save(embedding, join(EXPERIMENT_OUTPUT_PATH, 'embedding.pt'))
        with open(join(EXPERIMENT_OUTPUT_PATH, 'loss_history.pickle'), 'wb') as handle:
            pickle.dump(loss_history, handle)
        cumsum_vec = np.cumsum(np.insert(loss_history['total'], 0, 0)) 
        window_width = 10
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        plt.plot(ma_vec)
        save_figure(EXPERIMENT_NAME, 'convergence', 'png')
    else:
        embedding = torch.load(join(EXPERIMENT_OUTPUT_PATH, 'embedding.pt'))
    
    labeled_portions = np.round(np.linspace(0.1, 0.9, 9), 2)
    for labeled_portion in labeled_portions:
        print('Labeled portion:', labeled_portion, end=' ')
        _, microf1, macrof1 = dataset.get_evaluator().evaluate(embedding, 
                                    data['labels'], 
                                    labeled_portion=labeled_portion)
        print(f'Micro F1: {microf1:.2%} Macro F1: {macrof1:.2%}')