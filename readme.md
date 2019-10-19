# Implementations for PyTorch RNN tutorials with some additions.


Every script is GPU optimized unless it has "cpu" in its filename.

Replace nn.RNN with nn.GRU to get LSTM like performance. Keep in mind LSTM requires 2 hidden layers.


#### Data
Names from Pytorch tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
Thai names: https://hearnames.com/pronunciations/thai-names/thai-surnames
Translation pairs: https://www.manythings.org/anki/


#### Name classifiers.
1. `name_classifier_rnn_onehot.py` - Classify names using rnn with one hot encoded characters.
2. `name_classifier_rnn_embedding.py` - Classify names using rnn with dense embeddings.
3. `name_classifier_cnn_onehot.py` - Classify names using conv1d with one hot encoded characters.

#### Name generators.
1. `name_generator_cpu.py` - Implementation learning data char by char, resembling implementation in the PyTorch tutorial.
2. `name_generator_cpu_optimized.py` - Optimization of previous algorithm, learning from the whole name at once.
3. `name_generator.py` - GPU batched algorithm taking name prefixes and matching it with the next letter. Easy to understand implementation.
4. `name_generator_optimized.py` - Uses targets that are mapped to a packed sequence input. Uses less samples, therefore is faster and memory efficient.

#### Sentence to sequence translators.
1. `0.793` `seq2seq_forced_onehot.py` - Encoder output passed as initial hidden state of the decoder. Using teacher forcing and one hot encoded vectors.
2. `1.122` `seq2seq_unforced_onehot.py` - Encoder output passed as initial hidden state of the decoder. Tunable teacher forcing and one hot encoded vectors.
3. `0.700` `seq2seq_forced_propagated_onehot.py` - Encoder output is compressed and added to every input of the decoder. initial state of the decoder is empty. Using teacher forcing and one hot encoded vectors.
