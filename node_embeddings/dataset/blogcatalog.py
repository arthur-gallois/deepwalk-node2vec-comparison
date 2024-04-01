import numpy as np
import networkx as nx
from node_embeddings.dataset.dataset import Dataset
from node_embeddings.evaluators import SingleLabelEvaluator

class BlogCatalogDataset(Dataset):
    data_loc = 'data/BlogCatalog3/BlogCatalog-dataset/data/'

    def load(self):
        iid = {}
        idx = 0
        edgelist = []

        # Read edges pairs
        with open(self.data_loc+'edges.csv', 'r') as f:
            for line in f.readlines():
                i, j = line.strip().split(',')  # csv
                if i not in iid:
                    iid[i] = idx; idx += 1
                if j not in iid:
                    iid[j] = idx; idx += 1
                edgelist.append((iid[i], iid[j]))

        # Create an nx undirected network
        bc = nx.Graph(edgelist)

        print("Number of nodes: ", len(bc))
        print("Number of edges: ", bc.size())

        # Read labels
        labels = np.zeros((len(bc)), dtype=int)
        # Read (node_id, label) file
        with open(self.data_loc+'group-edges.csv', 'r') as f:
            for line in f.readlines():
                node, group = line.strip().split(',') 
                labels[iid[node]] = int(group)-1  

        bc_dataset = {'graph': bc, 'labels': labels}
        return bc_dataset
    
    def get_evaluator(self):
        return SingleLabelEvaluator()