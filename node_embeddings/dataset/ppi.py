import scipy
import torch
import numpy as np
import networkx as nx

from node_embeddings.dataset.dataset import Dataset


class PPIDataset(Dataset):
    data_loc = 'data/PPI/Homo_sapiens'

    def load(self):
        smat = scipy.io.loadmat(self.data_loc)
        adj_matrix, group = smat["network"], smat["group"]

        y = torch.from_numpy(group.todense()).to(torch.float)

        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)   

        dataset = {'graph': G, 'labels': y}

        return dataset