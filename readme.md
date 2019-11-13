# Practical, short, short, self-contained implementations of various NLP tasks using Pytorch 1.2.


Every script is GPU optimized unless it has "cpu" in its filename.


#### Data
Names of different nationalities are tanken from Pytorch tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
Thai names: https://hearnames.com/pronunciations/thai-names/thai-surnames
Translation pairs: https://www.manythings.org/anki/


#### Name classifiers.
1. `name_classifier_rnn_onehot.py`
  - Classify texts using RNN with one hot encoded vectors as inputs.
2. `name_classifier_rnn_embedding.py`
  - Classify names using RNN with dense embeddings.
3. `name_classifier_cnn_onehot.py` [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
  - Classify texts using CONV1D layers.
  - Using one hot encoded vectors.

#### Name generators.
1. `name_generator_cpu.py` - Implementation learning data char by char, resembling implementation in the PyTorch tutorial.
2. `name_generator_cpu_optimized.py` - Optimization of previous algorithm, learning from the whole name at once.
3. `name_generator.py` - GPU batched algorithm taking name prefixes and matching it with the next letter. Easy to understand implementation.
4. `name_generator_optimized.py` - Uses targets that are mapped to a packed sequence input. Uses less samples, therefore is faster and memory efficient.

#### Sentence to sequence translators.
1. `seq2seq_onehot.py` [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
  - Latest hidden state of the encoder RNN is passed as initial state of the decoder RNN.
  - This implementation is using teacher forcing and one hot encoded vectors.
2. `seq2seq_propagated_onehot.py` [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
  - Latest hidden state of the encoder RNN is passed as initial state of the decoder RNN.
  - Latest hidden state of the encoder RNN is added to every input of the decoder RNN.
  - Using teacher forcing and one hot encoded vectors.
3. `seq2seq_unforced_onehot.py` [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099)
  - Latest hidden state of the encoder RNN is passed as initial state of the decoder RNN.
  - Tunable teacher forcing for the decoder allows to use model outputs during training.
  - Using one hot encoded vectors.
4. `seq2seq_attn_onehot.py` [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
  - All hidden states of the encoder RNN is passed to the decoder RNN.
  - Decoder RNN choses what hidden state to use when generating output, therefore removing the bottleneck introduced by encoder output vector size.
  - Using one hot encoded vectors.
