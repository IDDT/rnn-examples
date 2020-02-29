import os
import time
import unicodedata
import string
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split



#%% Make charset.
BOS, EOS = '<', '>'
ALL_CHARS = BOS + string.ascii_letters + " .,;'-" + EOS
char_to_ix = {char:ix for ix, char in enumerate(ALL_CHARS)}
N_CHARS = len(char_to_ix)



#%% Make dataset.
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        names = set()
        for filename in os.listdir('data/names'):
            if filename.endswith('.txt'):
                with open('data/names/' + filename) as f:
                    for line in f:
                        names.add(BOS + self.to_ascii(line).strip() + EOS)
        self.names = list(names)

    @staticmethod
    def to_ascii(s):
        out = []
        for char in unicodedata.normalize('NFD', s):
            if unicodedata.category(char) != 'Mn' and char in ALL_CHARS:
                out.append(char)
        return ''.join(out)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, ix):
        return self.names[ix]

def make_x_y(documents:list):
    device = torch.device('cpu')

    #Make list of tensors.
    X = []
    for doc in documents:
        #Not taking the last char (EOS) for input.
        indices = [char_to_ix[char] for char in doc[0:-1]]
        X.append(torch.tensor(indices, dtype=torch.int64, device=device,
            requires_grad=False))

    #Make sorted packed sequence.
    X_lengths, X_indexes = torch.tensor([len(x) for x in X],
        dtype=torch.int16, device=device, requires_grad=False).sort(descending=True)
    X = nn.utils.rnn.pad_sequence(X, batch_first=True)[X_indexes]
    Z = nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

    #Place targets for NLLLoss according to the inputs in Z.data
    y = torch.zeros(len(Z.data), dtype=torch.int64, device=device, requires_grad=False)
    for i, ix in enumerate(X_indexes):
        #Not taking the first char (BOS) for targets.
        targets = [char_to_ix[char] for char in documents[ix][1:]]
        for t, target in enumerate(targets):
            y[i + Z.batch_sizes[0:t].sum()] = target

    return Z, y



#%% Init dataset.
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 256
dataset = Dataset()
n_test = len(dataset) // 20 // batch_size * batch_size
dataset_train, dataset_test = \
    random_split(dataset, (len(dataset) - n_test, n_test))
assert len(dataset_test) >= batch_size, "Batch size should be reduced."
dataloader_train = DataLoader(dataset_train, collate_fn=make_x_y,
    batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
dataloader_test = DataLoader(dataset_test, collate_fn=make_x_y,
    batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)



#%% Make model.
INPUT_SIZE = len(char_to_ix)
HIDDEN_SIZE = 100

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        emb_dim = 100
        self.emb = nn.Embedding(INPUT_SIZE, emb_dim)
        self.rnn = nn.GRU(emb_dim, HIDDEN_SIZE, batch_first=True)
    def forward(self, x):
        if type(x) is torch.nn.utils.rnn.PackedSequence:
            x = nn.utils.rnn.PackedSequence(self.emb(x.data),
                batch_sizes=x.batch_sizes, sorted_indices=x.sorted_indices,
                unsorted_indices=x.unsorted_indices)
        elif type(x) is torch.Tensor:
            assert len(x) == 1
            x = self.emb(x)
        else:
            raise ValueError()
        _, h = self.rnn(x)
        return h

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        emb_dim = 100
        self.emb = nn.Embedding(INPUT_SIZE, emb_dim)
        self.rnn = nn.GRU(emb_dim, HIDDEN_SIZE, batch_first=True)
        self.lin_o = nn.Linear(HIDDEN_SIZE, INPUT_SIZE)
    def forward(self, x, h):
        assert type(x) is torch.nn.utils.rnn.PackedSequence
        x = nn.utils.rnn.PackedSequence(self.emb(x.data),
            batch_sizes=x.batch_sizes, sorted_indices=x.sorted_indices,
            unsorted_indices=x.unsorted_indices)
        x, _ = self.rnn(x, h)
        x = self.lin_o(x.data)
        return F.log_softmax(x, dim=1)
    def predict(self, x, h):
        assert type(x) is torch.Tensor
        x = self.emb(x)
        x, h_n = self.rnn(x, h)
        x = self.lin_o(x)
        return F.log_softmax(x, dim=2), h_n


#%% Init model.
encoder = Encoder().to(device)
decoder = Decoder().to(device)
# model.load_state_dict(torch.load('models/name_vae.model', map_location=device))
loss_fn = nn.NLLLoss(reduction='mean')
optim = torch.optim.Adam((*encoder.parameters(), *decoder.parameters()), lr=0.001)



#%% Train.
loss_min = float('inf')
max_unimproved_epochs, unimproved_epochs = 15, 0
for epoch in range(1, 1000):
    start_time = time.time()
    #Training.
    encoder.train()
    decoder.train()
    losses = []
    for X, y in dataloader_train:
        X, y = X.to(device), y.to(device)
        h = encoder(X)
        y_pred = decoder(X, h)
        loss = loss_fn(y_pred, y)
        assert torch.isfinite(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    loss_train = sum(losses) / len(losses)
    #Testing.
    encoder.eval()
    decoder.eval()
    losses = []
    with torch.no_grad():
        for X, y in dataloader_test:
            X, y = X.to(device), y.to(device)
            h = encoder(X)
            y_pred = decoder(X, h)
            loss = loss_fn(y_pred, y)
            assert torch.isfinite(loss)
            losses.append(loss.item())
    loss_test = sum(losses) / len(losses)
    #Feedback.
    print(f'E{epoch} TRAIN:{loss_train:.5f} TEST:{loss_test:.5f}'
        f' TOOK:{time.time() - start_time:.1f}s')
    #Save state & early stopping.
    unimproved_epochs += 1
    if loss_test < loss_min:
        #torch.save(model.state_dict(), 'models/name_vae.model')
        loss_min = loss_test
        unimproved_epochs = 0
    if unimproved_epochs > max_unimproved_epochs:
        print(f'E{epoch} Early stopping. BEST TEST:{loss_min:.5f}')
        break



#%% Sample.
encoder.eval()
decoder.eval()
ix_to_char = {ix:char for char, ix in char_to_ix.items()}

def make_input(document:str) -> torch.tensor:
    indices = [char_to_ix[char] for char in document]
    return torch.tensor(indices, dtype=torch.int64, device=device,
        requires_grad=False)

def greedy_decode(hidden, max_length=100):
    out = '<'
    i = 0
    with torch.no_grad():
        while len(out) < max_length:
            #Make char vector.
            char_ix = char_to_ix[out[i]]
            char_vect = torch.tensor([[char_ix]], dtype=torch.int64,
                device=device, requires_grad=False)
            #Run prediction.
            probas, hidden = decoder.predict(char_vect, hidden)
            #Add prediction if last char.
            if i == len(out) - 1:
                topv, topi = probas.squeeze().topk(1)
                char = ix_to_char[topi[0].item()]
                out += char
                if char == '>':
                    break
            i += 1
    return out

h = encoder(make_input('<Orlov').unsqueeze(0))
greedy_decode(h, max_length=100)
