from os.path import join
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from node_embeddings.model import deepWalk
from node_embeddings.dataset import blogcatalog
from node_embeddings.experiments.base import save_figure, init

EXPERIMENT_NAME = 'basic_run_test'
EXPERIMENT_OUTPUT_PATH = join('output', EXPERIMENT_NAME)

if __name__ == '__main__':
    init(path=EXPERIMENT_OUTPUT_PATH, seed=1234)
    bc_dataset = blogcatalog.load_data()

    embedding, loss_history = deepWalk(
        graph=bc_dataset['graph'],  
        walks_per_vertex=1, 
        walk_length=10, 
        window_size=10,  
        embedding_size=128,
        num_neg=5,
        lr=1e-2,
        epochs=2,
        batch_size=50
    )
    torch.save(embedding, join(EXPERIMENT_OUTPUT_PATH, 'embedding.pt'))
    with open(join(EXPERIMENT_OUTPUT_PATH, 'loss_history.pickle'), 'wb') as handle:
        pickle.dump(bc_dataset, handle)

    cumsum_vec = np.cumsum(np.insert(loss_history['total'], 0, 0)) 
    window_width = 10
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    plt.plot(ma_vec)
    save_figure(EXPERIMENT_NAME, 'convergence', 'png')

    X = embedding.detach().numpy()
    y = bc_dataset['labels']

    clf = LogisticRegression(random_state=0, multi_class='ovr').fit(X, y)
    y_hat = clf.predict(X)
    print('Micro F1: ', f1_score(y, y_hat, average='micro'))
    print('Macro F1: ', f1_score(y, y_hat, average='macro'))