from os.path import join

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from node_embeddings.dataset import get_dataset, datasets

import argparse
import random

SEED=1234
np.random.seed(SEED)
random.seed(SEED)

classifiers = {
    'ppi': MultiOutputClassifier(estimator=LogisticRegression(random_state=0)),
    'blogcatalog': LogisticRegression(random_state=0, multi_class='ovr')
}

parser = argparse.ArgumentParser(prog='Split Experiment')
parser.add_argument('-d', '--dataset', choices=datasets.keys(), required=True)   
args = parser.parse_args()

dataset = args.dataset

DEEPWALK_EXPERIMENT_NAME = f'split_experiment_{dataset}_deepwalk'
NODE2VEC_EXPERIMENT_NAME = f'split_experiment_{dataset}_node2vec'
DEEPWALK_OUTPUT_PATH = join('output', DEEPWALK_EXPERIMENT_NAME)
NODE2VEC_OUTPUT_PATH = join('output', NODE2VEC_EXPERIMENT_NAME)
FINAL_OUTPUT_PATH = 'output'

def evaluate(embedding, labels, classifier, labeled_portion):
    X = embedding.detach().numpy()
    y = labels
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=(1-labeled_portion))

    clf = classifier.fit(X_train, y_train)
    
    y_hat = clf.predict(X_test)
    
    microf1 = f1_score(y_test, y_hat, average='micro')
    macrof1 = f1_score(y_test, y_hat, average='macro')
    return microf1, macrof1

def get_scores(labeled_portions, embedding):
    macro_f1_score = np.zeros(len(labeled_portions))
    micro_f1_score = np.zeros(len(labeled_portions))
    for i in range(len(labeled_portions)):
        print('Labeled portion:', labeled_portions[i], end=' ')
        microf1, macrof1 = evaluate(embedding, 
                                    bc_dataset['labels'], 
                                    classifier = classifier, 
                                    labeled_portion=labeled_portions[i])
        micro_f1_score[i] = microf1
        macro_f1_score[i] = macrof1
        print(f'Micro F1: {microf1:.2%}, Macro F1: {macrof1:.2%}')
    print()
    return micro_f1_score, macro_f1_score

def plot(filename, x_values, dw, n2v, title, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, dw, marker='o', linestyle='-', color='blue', label='DeepWalk')
    plt.plot(x_values, n2v, marker='o', linestyle='-', color='red', label='Node2Vec')

    plt.title(title)
    plt.xlabel('Labeled Portion')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    for i in range(len(dw)):
        plt.annotate(f"{dw[i]:.2%}", (x_values[i], dw[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f"{n2v[i]:.2%}", (x_values[i], n2v[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.savefig(join(FINAL_OUTPUT_PATH ,filename))
    plt.show()
    

if __name__ == '__main__':
    bc_dataset = get_dataset(dataset).load()
    classifier = classifiers[dataset]
    
    labeled_portions = np.arange(0.1, 1, 0.1).round(2)

    embedding_deepwalk = torch.load(join(DEEPWALK_OUTPUT_PATH, 'embedding.pt'))
    embedding_node2vec = torch.load(join(NODE2VEC_OUTPUT_PATH, 'embedding.pt'))

    print('\nEvaluating DeepWalk')
    micro_score_dw, macro_score_dw = get_scores(labeled_portions, embedding_deepwalk)
    print('Evaluating Node2Vec')
    micro_score_n2v, macro_score_n2v = get_scores(labeled_portions, embedding_node2vec)

    plot(f'{dataset}_micro.png',
         labeled_portions, 
         micro_score_dw, 
         micro_score_n2v, 
         f'Performance on {dataset}', 
         'Micro F1(%)'
    )

    plot(f'{dataset}_macro.png',
         labeled_portions, 
         macro_score_dw, 
         macro_score_n2v, 
         f'Performance on {dataset}', 
         'Macro F1(%)'
    )