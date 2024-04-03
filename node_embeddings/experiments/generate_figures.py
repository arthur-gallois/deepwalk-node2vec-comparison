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

parser = argparse.ArgumentParser(prog='Split Experiment')
parser.add_argument('-d', '--dataset', choices=datasets.keys(), required=True)   
args = parser.parse_args()

dataset_name = args.dataset

DEEPWALK_EXPERIMENT_NAME = f'split_experiment_{dataset_name}_deepwalk'
NODE2VEC_EXPERIMENT_NAME = f'split_experiment_{dataset_name}_node2vec'
DEEPWALK_OUTPUT_PATH = join('output', DEEPWALK_EXPERIMENT_NAME)
NODE2VEC_OUTPUT_PATH = join('output', NODE2VEC_EXPERIMENT_NAME)
FINAL_OUTPUT_PATH = 'output'

def get_scores(evaluator, labels, labeled_portions, embedding):
    NUM_RUNS = 10
    macro_f1_score = np.zeros(len(labeled_portions))
    micro_f1_score = np.zeros(len(labeled_portions))
    for i in range(len(labeled_portions)):
        print('Labeled portion:', labeled_portions[i], end=' ')
        for _ in range(NUM_RUNS):
            _, microf1, macrof1 = evaluator.evaluate(embedding, 
                                        labels, 
                                        labeled_portion=labeled_portions[i])
            micro_f1_score[i] += microf1
            macro_f1_score[i] += macrof1
        print(f'Micro F1: {micro_f1_score[i]/NUM_RUNS:.2%}, Macro F1: {macro_f1_score[i]/NUM_RUNS:.2%}')
    print()
    return micro_f1_score/NUM_RUNS, macro_f1_score/NUM_RUNS

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
        plt.annotate(f"{dw[i]:.2%}", (x_values[i], dw[i]), textcoords="offset points", xytext=(0,-10), ha='center')
        plt.annotate(f"{n2v[i]:.2%}", (x_values[i], n2v[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.savefig(join(FINAL_OUTPUT_PATH ,filename))
    plt.show()
    

if __name__ == '__main__':
    dataset = get_dataset(dataset_name)
    data = dataset.load()
    
    labeled_portions = np.arange(0.1, 1, 0.1).round(2)

    embedding_deepwalk = torch.load(join(DEEPWALK_OUTPUT_PATH, 'embedding.pt'))
    embedding_node2vec = torch.load(join(NODE2VEC_OUTPUT_PATH, 'embedding.pt'))

    print('\nEvaluating DeepWalk')
    micro_score_dw, macro_score_dw = get_scores(dataset.get_evaluator(), 
                                                data['labels'], 
                                                labeled_portions, 
                                                embedding_deepwalk)
    print('Evaluating Node2Vec')
    micro_score_n2v, macro_score_n2v = get_scores(dataset.get_evaluator(), 
                                                data['labels'], 
                                                labeled_portions, 
                                                embedding_node2vec)
    plot(f'{dataset_name}_micro.svg',
         labeled_portions, 
         micro_score_dw, 
         micro_score_n2v, 
         f'Performance on {dataset_name}', 
         'Micro F1(%)'
    )

    plot(f'{dataset_name}_macro.svg',
         labeled_portions, 
         macro_score_dw, 
         macro_score_n2v, 
         f'Performance on {dataset_name}', 
         'Macro F1(%)'
    )