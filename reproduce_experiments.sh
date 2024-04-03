python -m node_embeddings.experiments.split_experiment --dataset ppi \
    --algorithm deepwalk \
    --walks_per_vertex 40 \
    --walk_length 80 \
    --window_size 10 \
    --embedding_size 128 \
    --num_negatives 2 \
    --learning_rate 1e-1 \
    --epochs 1 \
    --batch_size 64  

python -m node_embeddings.experiments.split_experiment --dataset ppi \
    --algorithm node2vec \
    --walks_per_vertex 40 \
    --walk_length 80 \
    --window_size 10 \
    --embedding_size 128 \
    --num_negatives 2 \
    --learning_rate 1e-1 \
    --epochs 1 \
    --batch_size 64 \
    -p 4 -q 1

python -m node_embeddings.experiments.generate_figures -d ppi

python -m node_embeddings.experiments.split_experiment --dataset blogcatalog \
    --algorithm deepwalk \
    --walks_per_vertex 80 \
    --walk_length 80 \
    --window_size 10 \
    --embedding_size 128 \
    --num_negatives 2 \
    --learning_rate 1e-1 \
    --epochs 1 \
    --batch_size 64  

python -m node_embeddings.experiments.split_experiment --dataset blogcatalog \
    --algorithm node2vec \
    --walks_per_vertex 80 \
    --walk_length 80 \
    --window_size 10 \
    --embedding_size 128 \
    --num_negatives 2 \
    --learning_rate 1e-1 \
    --epochs 1 \
    --batch_size 64 \
    -p 0.25 -q 0.25

python -m node_embeddings.experiments.generate_figures -d blogcatalog