# Practical, short, batched, gpu-ready, self-contained implementations of various machine learning tasks using Pytorch 1.3.0


#### Data
Names of different nationalities are taken from Pytorch tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
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
1. `name_generator.py`
  - Taking name prefixes and matching it with the next letter. Easy to understand implementation.
2. `name_generator_optimized.py`
  - Targets are mapped to input packed sequence order. Uses less samples, therefore is faster and memory efficient.

#### Sentence to sequence translators.
1. `seq2seq_onehot.py` [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
  - Latest hidden state of the encoder RNN is passed as initial state of the decoder RNN.
  - This implementation is using teacher forcing and one hot encoded vectors.
2. `seq2seq_unforced_onehot.py` [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099)
  - Latest hidden state of the encoder RNN is passed as initial state of the decoder RNN.
  - Tunable teacher forcing for the decoder allows to use model outputs during training.
  - Using one hot encoded vectors.
3. `seq2seq_propagated_onehot.py` [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
  - Latest hidden state of the encoder RNN is passed as initial state of the decoder RNN as well as additional context to be combined with every hidden state and output.
  - Using teacher forcing and one hot encoded vectors.
4. `seq2seq_attn_onehot.py` [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
  - All hidden states of the encoder RNN is passed to the decoder RNN.
  - Decoder RNN choses what hidden state to use when generating output, therefore removing the bottleneck introduced by encoder output vector size.
  - Using one hot encoded vectors.

#### Variational autoencoders.
1. `vae_mnist.py`
  - Reconstruct MNIST images through fully connected network.
2. `vae_conv_mnist.py`
  - Reconstruct MNIST images through layered convolutions.
