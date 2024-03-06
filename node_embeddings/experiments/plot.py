import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from pathlib import Path

sns.set()

OUTPUT_FOLDER = 'output'

def save_figure(experiment_name, figure_name, format):
    plt.savefig(
        join(OUTPUT_FOLDER, experiment_name, figure_name+'.'+format), 
        format=format, 
        dpi=1200
    )

def create_directory(experiment_path):
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
