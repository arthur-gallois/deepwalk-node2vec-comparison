from node_embeddings.dataset.blogcatalog import BlogCatalogDataset
from node_embeddings.dataset.ppi import PPIDataset

datasets = {
    'ppi': PPIDataset,
    'blogcatalog': BlogCatalogDataset
}

def get_dataset(dataset):
    return datasets[dataset]()