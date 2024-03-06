# DeepWalk x Node2Vec

Authors: Alexandre Maranhão, Alex Pierron and Arthur Gallois 

This repository is dedicated to implementing DeepWalk, Node2Vec and comparing their results.

### Setup 

To setup the environment, install the requirements from `requirements.txt` using the environment manager of your choice. One possibility is with venv, by executing

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Experiments

Download the dataset with

```bash
sh download_blogcatalog.sh
```

To run the experiments, execute from the repository root

```bash
python -m node_embeddings.experiments.basic_run
```

The output images will be written in a folder called `output` under the repository root.

### References

- *Bryan Perozzi, Rami Al-Rfou, and Steven Skiena* (2014). **DeepWalk: online learning of social representations**. In: Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '14). Association for Computing Machinery, New York, NY, USA, 701–710. [![DOI:https://doi.org/10.1145/2623330.2623732](https://zenodo.org/badge/DOI/10.1145/2623330.2623732.svg)](https://doi.org/10.1145/2623330.2623732)
- *Aditya Grover and Jure Leskovec* (2016). **Node2vec: Scalable Feature Learning for Networks**. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16). Association for Computing Machinery, New York, NY, USA, 855–864. [![DOI:https://doi.org/10.1145/2939672.2939754](https://zenodo.org/badge/DOI/10.1145/2939672.2939754.svg)](https://doi.org/10.1145/2939672.2939754)