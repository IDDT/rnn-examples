import os
import time
import unicodedata
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



#GPU ready classifier with large batch using packed sequence.
#RNN on one-hot encoded char vectors.



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



#%% Load data.
ALL_CHARS = string.ascii_letters + " .,;'"

def normalize_string(string):
    out = []
    for char in unicodedata.normalize('NFD', string.strip()):
        if unicodedata.category(char) != 'Mn' and char in ALL_CHARS:
            out.append(char)
    return ''.join(out)

names, categories = [], []
for filename in os.listdir('data/names'):
    category = filename.strip('.txt').lower()
    if filename.endswith('.txt'):
        with open('data/names/' + filename) as f:
            for line in f:
                names.append(normalize_string(line))
                categories.append(category)

rand_idx = np.random.RandomState(seed=0).permutation(len(names))
names = np.array(names)[rand_idx.tolist()]
categories = np.array(categories)[rand_idx]

char_to_ix = {x: i for i, x in enumerate(ALL_CHARS)}
cate_to_ix = {x: i for i, x in enumerate(set(categories))}



#%% Split into train and test.
n_test = len(names) // 10
names, names_t = names[0:-n_test], names[-n_test:]
categories, categories_t = categories[0:-n_test], categories[-n_test:]



#%% Make X, y
input_size = len(char_to_ix)
output_size = len(cate_to_ix)

def make_input_vect(name):
    #for each sequence
    #   for each timestep
    #       for each feature
    vect = torch.zeros(len(name), len(char_to_ix), dtype=torch.float32,
        device=device, requires_grad=False)
    for c, char in enumerate(name):
        vect[c][char_to_ix[char]] = 1
    return vect

X = [make_input_vect(x) for x in names]
X_lengths, X_indexes = torch.tensor([len(x) for x in X],
    dtype=torch.int16, device=device, requires_grad=False).sort(descending=True)
X = nn.utils.rnn.pad_sequence(X, batch_first=True)[X_indexes]
Z = nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
y = torch.tensor([cate_to_ix[x] for x in categories],
    dtype=torch.int64, device=device, requires_grad=False)[X_indexes]

X = [make_input_vect(x) for x in names_t]
X_lengths, X_indexes = torch.tensor([len(x) for x in X],
    dtype=torch.int16, device=device, requires_grad=False).sort(descending=True)
X = nn.utils.rnn.pad_sequence(X, batch_first=True)[X_indexes]
Z_t = nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
y_t = torch.tensor([cate_to_ix[x] for x in categories_t],
    dtype=torch.int64, device=device, requires_grad=False)[X_indexes]



#%% Make category weights.
indices, counts = y.unique(return_counts=True)
indices, counts = indices.detach().tolist(), counts.detach().tolist()
weights = torch.zeros(len(cate_to_ix), dtype=torch.float32,
    device=device, requires_grad=False)
for ix, count in zip(indices, counts):
    weights[ix] = count
weights = weights.max() / weights.clamp(min=1)



#%% Model.
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        if type(x) is nn.utils.rnn.PackedSequence:
            batch_size = x.batch_sizes[0].item()
        elif type(x) is torch.Tensor:
            batch_size = x.shape[0]
        else:
            ValueError('Unknown input type.')
        h = torch.zeros(1, batch_size, self.hidden_size,
            dtype=torch.float32, device=device, requires_grad=True)
        _, h = self.rnn(x, h)
        h = self.lin(F.relu(h.reshape(-1, self.hidden_size)))
        return F.log_softmax(h, dim=1)



#%% Training.
hidden_size = 64
model = RNNClassifier(input_size, hidden_size, output_size).to(device)
loss_fn = nn.NLLLoss(weight=weights, reduction='mean')
loss_fn_test = nn.NLLLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

max_unimproved_epochs, unimproved_epochs = 50, 0
loss_min = np.inf
start_time = time.time()
for epoch in range(1001):
    #Training.
    model.train()
    y_pred = model(Z)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_train = loss.item()
    #Testing.
    model.eval()
    y_pred = model(Z_t)
    loss_test = loss_fn_test(y_pred, y_t).item()
    #Feedback.
    if epoch % 10 == 0:
        print(f'E {epoch} TRAIN: {loss_train:.3f} TEST: {loss_test:.3f}')
    #Early stopping.
    unimproved_epochs += 1
    if loss_test < loss_min:
        loss_min = loss_test
        unimproved_epochs = 0
    if unimproved_epochs > max_unimproved_epochs:
        minutes_took = (time.time() - start_time) / 60
        print(f'E {epoch} Early stopping. BEST TEST: {loss_min:.3f}')
        print(f'Took: {minutes_took:.1f}m')
        break



#%% Predict.
ix_to_cate = {ix: cate for cate, ix in cate_to_ix.items()}
model.eval()
y_pred = model(make_input_vect('Xiao').reshape(1, -1, len(char_to_ix)))
values, indexes = torch.exp(y_pred).topk(5)
values, indexes = values.reshape(-1), indexes.reshape(-1)
for i in range(len(values)):
    value, index = values[i].item(), indexes[i].item()
    print(f'{value * 100:.1f}% - {ix_to_cate[index]}')
