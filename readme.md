# Implementations for PyTorch RNN tutorials with some additions.


Every script is GPU optimized unless it has "cpu" in its filename.


#### Name classifiers.
1. `name_classifier_onehot.py` - Classify names using one hot encoded characters.
2. `name_classifier_embedding.py` - The same thing, but using dense vectors with "embedding" layer.
3. `name_classifier_cnn_onehot.py` - Classify names using one hot encoded vectors with 1D CNN.

#### Name generators.
1. `name_generator_cpu.py` - Implementation learning data char by char, resembling implementation in the PyTorch tutorial.
2. `name_generator_cpu_optimized.py` - Optimization of previous algorithm, learning from the whole name at once.
3. `name_generator_gpu.py` - GPU batched algorithm taking name prefixes and matching it with the next letter. Easy to understand.
4. `name_generator_gpu_optimized.py` - Uses targets that are mapped to packed sequence input. Uses less samples, hence faster and memory efficient.
