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
HIDDEN_SIZE = 250
Z_DIM = 2

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = HIDDEN_SIZE
        emb_dim = 100
        #Shared embedding.
        self.emb = nn.Embedding(INPUT_SIZE, emb_dim)
        #Encoder layers.
        self.rnn_enc = nn.GRU(emb_dim, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.lin_mean = nn.Linear(HIDDEN_SIZE * 2, Z_DIM)
        self.lin_log_sigma = nn.Linear(HIDDEN_SIZE * 2, Z_DIM)
        #Decoder layers.
        self.lin_h = nn.Linear(Z_DIM, HIDDEN_SIZE)
        self.rnn_dec = nn.GRU(emb_dim, HIDDEN_SIZE, batch_first=True)
        self.lin_o = nn.Linear(HIDDEN_SIZE, INPUT_SIZE)

    def encode(self, x):
        assert type(x) == torch.nn.utils.rnn.PackedSequence
        batch_size = x.batch_sizes[0]
        x = nn.utils.rnn.PackedSequence(self.emb(x.data),
            batch_sizes=x.batch_sizes, sorted_indices=x.sorted_indices,
            unsorted_indices=x.unsorted_indices)
        _, h = self.rnn_enc(x)
        h = h.reshape(batch_size, -1)
        z_mean = self.lin_mean(h)
        z_log_sigma = self.lin_log_sigma(h)
        return z_mean, z_log_sigma

    @staticmethod
    def sample_z(z_mean, z_log_sigma):
        epsilon = torch.randn_like(z_log_sigma)
        return z_mean + torch.exp(z_log_sigma / 2) * epsilon

    def decode(self, z, x):
        assert type(x) == torch.nn.utils.rnn.PackedSequence
        assert z.shape[0] == x.batch_sizes[0]
        x = nn.utils.rnn.PackedSequence(self.emb(x.data),
            batch_sizes=x.batch_sizes, sorted_indices=x.sorted_indices,
            unsorted_indices=x.unsorted_indices)
        h = self.lin_h(z).unsqueeze(0)
        x, _ = self.rnn_dec(x, h)
        x = self.lin_o(x.data)
        return F.log_softmax(x, dim=1)

    def forward(self, x):
        z_mean, z_log_sigma = self.encode(x)
        z = self.sample_z(z_mean, z_log_sigma)
        y_pred = self.decode(z, x)
        return y_pred, z_mean, z_log_sigma

    def predict(self, x, h):
        #Output and hidden are the same if seq_len = 1.
        x = self.emb(x)
        _, h = self.rnn_dec(x, h)
        probas = F.softmax(self.lin_o(h), dim=2)
        return probas, h

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.NLLLoss(reduction='mean')
    def forward(self, y_pred, y, z_mean, z_log_sigma):
        nll_loss = self.loss_fn(y_pred, y)
        kl_loss = 0.5 * (z_log_sigma.exp() + z_mean.pow(2) - 1 - z_log_sigma).sum(axis=1)
        #print(f'{nll_loss.mean().item():.4f}, {kl_loss.mean().item():.4f}')
        return (nll_loss + kl_loss).mean()



#%% Init model.
model = VAE().to(device)
# model.load_state_dict(torch.load('models/name_vae.model', map_location=device))
loss_fn = VAELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)



#%% Train.
loss_min = float('inf')
max_unimproved_epochs, unimproved_epochs = 15, 0
for epoch in range(1, 1000):
    start_time = time.time()
    #Training.
    model.train()
    losses = []
    for X, y in dataloader_train:
        X, y = X.to(device), y.to(device)
        y_pred, z_mean, z_log_sigma = model(X)
        loss = loss_fn(y_pred, y, z_mean, z_log_sigma)
        assert torch.isfinite(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    loss_train = sum(losses) / len(losses)
    #Testing.
    model.eval()
    losses = []
    z_mean_arr = []
    with torch.no_grad():
        for X, y in dataloader_test:
            X, y = X.to(device), y.to(device)
            y_pred, z_mean, z_log_sigma = model(X)
            loss = loss_fn(y_pred, y, z_mean, z_log_sigma)
            assert torch.isfinite(loss)
            losses.append(loss.item())
            z_mean_arr += z_mean.tolist()
    loss_test = sum(losses) / len(losses)
    #Feedback.
    print(f'E{epoch} TRAIN:{loss_train:.5f} TEST:{loss_test:.5f}'
        f' TOOK:{time.time() - start_time:.1f}s')
    plt.scatter(*np.hsplit(np.array(z_mean_arr), 2), s=0.5)
    plt.xlim((-0.25, 0.25))
    plt.ylim((-0.25, 0.25))
    plt.show()
    #Save state & early stopping.
    unimproved_epochs += 1
    if loss_test < loss_min:
        torch.save(model.state_dict(), 'models/name_vae.model')
        loss_min = loss_test
        unimproved_epochs = 0
    if unimproved_epochs > max_unimproved_epochs:
        print(f'E{epoch} Early stopping. BEST TEST:{loss_min:.5f}')
        break



#%% Print distribution.
model.eval()
z_mean_arr, z_log_sigma_arr = [], []
with torch.no_grad():
    for X, y in dataloader_test:
        X, y = X.to(device), y.to(device)
        z_mean, _ = model.encode(X)
        z_mean_arr += z_mean.tolist()
        z_log_sigma_arr += z_log_sigma.tolist()


plt.scatter(*np.hsplit(np.array(z_mean_arr), 2), s=0.5)

plt.scatter(*np.hsplit(np.array(z_log_sigma_arr), 2), s=0.5)



#%% Sample.
model.eval()
ix_to_char = {ix:char for char, ix in char_to_ix.items()}

def make_input(document:str) -> torch.tensor:
    indices = [char_to_ix[char] for char in document]
    return torch.tensor(indices, dtype=torch.int64, device=device,
        requires_grad=False)

def greedy_decode(hidden, max_length=100):
    hidden = decoder.lin_h(hidden)
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
                topv, topi = probas.topk(1)
                char = ix_to_char[topi[0].item()]
                out += char
                if char == '>':
                    break
            i += 1
    return out

with torch.no_grad():
    z_mean, z_log_sigma = encoder(make_input('<Daniel').unsqueeze(0))
    hidden = model.sample_z(z_mean, z_log_sigma).unsqueeze(0)
greedy_decode(hidden, max_length=100)


greedy_decode(torch.tensor([[[6.0, -7.5]]]).to(device))
hidden
