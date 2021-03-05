import time
import os
import re
import unicodedata
import string

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import axes3d



CHAR_START, CHAR_END = '<', '>'
ALL_CHARS = CHAR_START + string.ascii_lowercase + " .,;'-" + CHAR_END
N_CHARS = len(ALL_CHARS)

#Init dataset.
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.device = torch.device('cpu')
        self.names = []
        self.categories = []
        for filename in os.listdir('data/names'):
            category = filename.strip('.txt').lower()
            if filename.endswith('.txt'):
                with open('data/names/' + filename) as f:
                    for line in f:
                        name = self.to_ascii(line.strip().lower())
                        if len(name) > 1:
                            self.names.append(CHAR_START + name + CHAR_END)
                            self.categories.append(category)

        self.char_to_ix = {x: i for i, x in enumerate(ALL_CHARS)}
        self.ix_to_char = {i: x for i, x in enumerate(ALL_CHARS)}

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
        return torch.tensor([self.char_to_ix[char] for char in self.names[ix]],
            dtype=torch.int64, device=self.device, requires_grad=False), self.categories[ix]

    def make_batch(self, batch):
        device = torch.device('cpu')
        indices, label = zip(*batch)

        X_len, X_idx = torch.tensor([len(idx) for idx in indices],
            dtype=torch.int16, device=device, requires_grad=False)\
            .sort(descending=True)
        X = nn.utils.rnn.pad_sequence([i[1:-1] for i in indices],
            batch_first=True)[X_idx]
        Z_i = nn.utils.rnn.pack_padded_sequence(X, X_len-2, batch_first=True)

        X = nn.utils.rnn.pad_sequence([i[:-1] for i in indices],
            batch_first=True)[X_idx]
        Z_o = nn.utils.rnn.pack_padded_sequence(X, X_len-1, batch_first=True)

        X = nn.utils.rnn.pad_sequence([i[1:] for i in indices],
            batch_first=True)[X_idx]
        Z = nn.utils.rnn.pack_padded_sequence(X, X_len-1, batch_first=True)

        return Z_i, Z_o, Z.data, label



#Settings.
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 128

dataset = Dataset()
n_test = len(dataset) // 10
dataset_train, dataset_test = \
    random_split(dataset, (len(dataset) - n_test, n_test))
assert len(dataset_test) >= batch_size, "Batch size should be reduced."
dataloader_train = DataLoader(dataset_train, collate_fn=dataset.make_batch,
    batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
dataloader_test = DataLoader(dataset_test, collate_fn=dataset.make_batch,
    batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)



#Init model.
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, z_dim):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.lin_mean = nn.Linear(hidden_size*2, z_dim)
        self.lin_log_var = nn.Linear(hidden_size*2, z_dim)

    def forward(self, x):
        if type(x) == nn.utils.rnn.PackedSequence:
            x = nn.utils.rnn.PackedSequence(
                data=self.emb(x.data), batch_sizes=x.batch_sizes)
        elif type(x) == torch.Tensor:
            x = self.emb(x)
        else:
            raise ValueError('Unknown tensor type.')

        h = self.rnn(x)[1].transpose(0,1).reshape(-1, self.hidden_size*2)

        z_mean = self.lin_mean(h)
        z_log_var = self.lin_log_var(h)
        return z_mean, z_log_var

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lin_o = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        if type(x) == nn.utils.rnn.PackedSequence:
            x = nn.utils.rnn.PackedSequence(
                data=self.emb(x.data), batch_sizes=x.batch_sizes)
        elif type(x) == torch.Tensor:
            x = self.emb(x)
        else:
            raise ValueError('Unknown tensor type.')

        out, _ = self.rnn(x, h)

        return F.log_softmax(self.lin_o(out.data), dim=1)

    def predict(self, x, h):
        x = self.emb(x)
        _, h = self.rnn(x, h)
        return F.softmax(self.lin_o(h), dim=2), h

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size=128, z_dim=128):
        super().__init__()
        self.lin_h = nn.Linear(z_dim, hidden_size)
        self.encoder = Encoder(input_size, hidden_size, z_dim)
        self.decoder = Decoder(hidden_size, input_size)
        self.hidden_size = hidden_size

    @staticmethod
    def sample_z(z_mean, z_log_var):
        epsilon = torch.randn_like(z_log_var)
        return z_mean + torch.exp(z_log_var / 2) * epsilon

    def forward(self, x_i, x_o):
        #Encode, sample from distribution, decode.
        z_mean, z_log_var = self.encoder(x_i)
        h_pred = self.sample_z(z_mean, z_log_var)
        h_pred = self.lin_h(h_pred).unsqueeze(0)
        return self.decoder(x_o, h_pred), z_mean, z_log_var

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_pred, x, z_mean, z_log_var):
        r_loss = F.nll_loss(x_pred, x, reduction='mean')
        kl_loss = (z_log_var.exp() + z_mean.pow(2) - 1 - z_log_var).mean(axis=1)
        mean_loss = (r_loss + kl_loss).mean()

        return r_loss.mean(), kl_loss.mean(), \
            (r_loss*r_loss + kl_loss*kl_loss).mean() / mean_loss



loss_fn = VAELoss()
model = VAE(len(dataset.char_to_ix)).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)



#%% Train.
max_unimproved_epochs, unimproved_epochs = 5, 0
loss_min = float('inf')
for epoch in range(1000):
    start_time = time.time()
    #Training.
    model.train()
    losses = []
    for X_i, X_o, y, _ in dataloader_train:
        optim.zero_grad()
        X_pred, z_mean, z_log_var = model(X_i, X_o)
        _, _, loss = loss_fn(X_pred, y, z_mean, z_log_var)
        loss.backward()
        losses.append(loss.item())
        optim.step()
    loss_train = sum(losses) / len(losses)
    #Testing.
    model.eval()
    losses = []
    klds = []
    nll_losses = []
    with torch.no_grad():
        for X_i, X_o, y, _ in dataloader_test:
            X_pred, z_mean, z_log_var = model(X_i, X_o)
            nll_loss, kld, loss = loss_fn(X_pred, y, z_mean, z_log_var)
            losses.append(loss.item())
            nll_losses.append(nll_loss.item())
            klds.append(kld.item())
    loss_test = sum(losses) / len(losses)
    kld_test = sum(klds) / len(klds)
    nll_test = sum(nll_losses) / len(nll_losses)
    #Feedback.
    print(f'E {epoch} TRAIN: {loss_train:.3f} TEST: {loss_test:.3f}'
        f' KLD: {kld_test:.3f} NLL: {nll_test:.3f}'
        f' TOOK: {time.time() - start_time:.1f}s')
    #Save state & early stopping.
    unimproved_epochs += 1
    if loss_test < loss_min:
        torch.save(model.state_dict(), 'models/vae_name_generator.model')
        torch.save(optim.state_dict(), 'models/vae_name_generator.optim')
        loss_min = loss_test
        unimproved_epochs = 0
    if unimproved_epochs > max_unimproved_epochs:
        print(f'E {epoch} Early stopping. BEST TEST: {loss_min:.3f}')
        break

if __name__ == "__main__":
    quit()


model = VAE(len(dataset.char_to_ix)).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load('models/vae_name_generator.model', map_location=device))
optim.load_state_dict(torch.load('models/vae_name_generator.optim', map_location=device))

model.eval()
model.to(torch.device('cpu'))

def predict(name, MAX_LEN=50, model=model, dataset=dataset):
    device = torch.device('cpu')
    input = torch.tensor([dataset.char_to_ix[char] for char in name.lower()])

    h, _ = model.encoder(input.unsqueeze(0)) #1,64
    h = model.lin_h(h).unsqueeze(0) #1, 1, 128
    x = torch.tensor([[dataset.char_to_ix[CHAR_START]]],
        dtype=torch.int64, device=device, requires_grad=False)

    out = CHAR_START
    while True:
        print(x)
        x, h = model.decoder.predict(x, h)
        x = x.argmax(dim=2)
        out += dataset.ix_to_char[x.item()]
        if out[-1] == CHAR_END or len(out) > MAX_LEN:
            break
    return out

def predict_range(name1, name2, MAX_LEN=50, model=model, dataset=dataset):
    device = torch.device('cpu')

    input1 = torch.tensor([dataset.char_to_ix[char] for char in name1.lower()])
    input2 = torch.tensor([dataset.char_to_ix[char] for char in name2.lower()])

    h1, _ = model.encoder(input1.unsqueeze(0)) #1,64
    h2, _ = model.encoder(input2.unsqueeze(0)) #1,64

    all_outs = []
    for a in np.arange(0,1.1,0.1):
        h = a*h1 + (1 - a)*h2
        h = model.lin_h(h).unsqueeze(0) #1, 1, 128
        x = torch.tensor([[dataset.char_to_ix[CHAR_START]]],
            dtype=torch.int64, device=device, requires_grad=False)

        out = CHAR_START
        while True:
            x, h = model.decoder.predict(x, h)
            x = x.argmax(dim=2)
            out += dataset.ix_to_char[x.item()]
            if out[-1] == CHAR_END or len(out) > 20:
                break

        all_outs.append(out)
    return all_outs

predict('kirill')
predict_range('kirill', 'orlov')

#%% Show distribution.
model.eval()
z_mean_arr, labels_arr = [], []
with torch.no_grad():
    for i, (X_i, _, _, label) in enumerate(dataloader_test):
        X_i = X_i.to(device)
        z_mean, _ = model.encoder(X_i)
        z_mean_arr.append(z_mean)
        labels_arr.extend(label)

z_mean = torch.cat(z_mean_arr, dim=0).cpu()
country_to_ix = {c:i for i, c in enumerate(set(labels_arr))}
labels = [country_to_ix[c] for c in labels_arr]

# #%%Plot scatter PCA
pca = PCA(n_components=2, whiten=True).fit(z_mean)
z_mean = pca.transform(z_mean)
# z_mean = TSNE(n_components=3).fit_transform(z_mean)
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='plasma', s=1)
plt.savefig("z_dim_128_kld_ratio.png", dpi=400)


#Quit here if ran as script.
if __name__ == '__main__':
    quit()
