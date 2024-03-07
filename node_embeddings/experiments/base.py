import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from pathlib import Path
import numpy as np
import torch
import random

OUTPUT_FOLDER = 'output'

def save_figure(experiment_name, figure_name, format):
    plt.savefig(
        join(OUTPUT_FOLDER, experiment_name, figure_name+'.'+format), 
        format=format, 
        dpi=1200
    )

def init(path, seed):
    sns.set()
    Path(path).mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
