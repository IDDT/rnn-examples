import os
import time
import unicodedata
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



#GPU ready classifier with large batch.
#1D CNN on one-hot encoded char vectors.



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



#Make X, y.
seq_len = max([len(name) for name in names])
input_size, output_size = len(char_to_ix), len(cate_to_ix)

def make_input_vectors(texts, char_to_ix, seq_len):
    #for each sequence
    #   for each feature
    #       for each timestep
    arr = torch.zeros(len(texts), len(char_to_ix), seq_len,
        dtype=torch.float32, device=device, requires_grad=False)
    for t, text in enumerate(texts):
        for c, char in enumerate(text):
            ix = char_to_ix[char]
            arr[t][ix][c] = 1
    return arr

X = make_input_vectors(names, char_to_ix, seq_len)
y = torch.tensor([cate_to_ix[x] for x in categories],
    dtype=torch.int64, device=device, requires_grad=False)
X_t = make_input_vectors(names_t, char_to_ix, seq_len)
y_t = torch.tensor([cate_to_ix[x] for x in categories_t],
    dtype=torch.int64, device=device, requires_grad=False)



#%% Make category weights.
counts = {}
for target in y_t:
    target = target.item()
    if target not in counts:
        counts[target] = 0
    counts[target] += 1
weights = torch.tensor([counts.get(i, 0) for i in range(len(cate_to_ix))],
    dtype=torch.float32, device=device, requires_grad=False)
weights = weights.max() / weights



#%% Model.
class CNNClassifier(nn.Module):
    def __init__(self, seq_len, input_size, output_size):
        super(CNNClassifier, self).__init__()
        self.conv_a = torch.nn.Conv1d(input_size, 20, 5)
        self.conv_b = torch.nn.Conv1d(input_size, 20, 4)
        self.conv_c = torch.nn.Conv1d(input_size, 20, 3)
        #Max value from all outputs.
        self.lin = nn.Linear(60, output_size)
    def forward(self, x):
        #batch, n_channels, seq_len, input_size = x.shape
        batch_size = x.shape[0]
        x = torch.cat((
            self.conv_a(x).max(dim=2)[0],
            self.conv_b(x).max(dim=2)[0],
            self.conv_c(x).max(dim=2)[0]
        ), dim=1)
        x = self.lin(F.relu(x))
        return F.log_softmax(x, dim=1)



#%% Training.
batch_size, input_size, seq_len = X.shape
model = CNNClassifier(seq_len, input_size, output_size).to(device)
loss_fn = nn.NLLLoss(weight=weights, reduction='mean')
loss_fn_test = nn.NLLLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

max_unimproved_epochs, unimproved_epochs = 50, 0
loss_min = np.inf
start_time = time.time()
for epoch in range(1001):
    #Training.
    model.train()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_train = loss.item()
    #Testing.
    model.eval()
    y_pred = model(X_t)
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
y_pred = model(make_input_vectors(['Xiao'], char_to_ix, seq_len)).reshape(-1)
values, indexes = torch.exp(y_pred).topk(5)
for i in range(len(values)):
    value, index = values[i].item(), indexes[i].item()
    print(f'{value * 100:.1f}% - {ix_to_cate[index]}')
