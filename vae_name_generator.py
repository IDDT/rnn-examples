import time
import os
import re
import unicodedata
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



#%% Settings.
CHAR_START, CHAR_END = '<', '>'
ALL_CHARS = CHAR_START + string.ascii_lowercase + " .,;'-" + CHAR_END


#%% Init dataset.
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.device = torch.device('cpu')
        self.names = []
        for filename in os.listdir('data/names'):
            category = filename.strip('.txt').lower()
            if filename.endswith('.txt'):
                with open('data/names/' + filename) as f:
                    for line in f:
                        name = self.to_ascii(line.strip().lower())
                        if len(name) > 1:
                            self.names.append(CHAR_START + name + CHAR_END)

        self.char_to_ix = {x: i for i, x in enumerate(ALL_CHARS)}
        self.ix_to_char = {i: x for i, x in enumerate(ALL_CHARS)}

    @staticmethod
    def to_ascii(s:str) -> str:
        out = []
        for char in unicodedata.normalize('NFD', s):
            if unicodedata.category(char) != 'Mn' and char in ALL_CHARS:
                out.append(char)
        return ''.join(out)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, ix):
        return torch.tensor([self.char_to_ix[char] for char in self.names[ix]],
            dtype=torch.int64, device=self.device, requires_grad=False)

    @staticmethod
    def make_batch(inputs):
        device = torch.device('cpu')
        X_len, X_idx = torch.tensor([len(idx) for idx in inputs],
            dtype=torch.int16, device=device, requires_grad=False)\
            .sort(descending=True)
        X = nn.utils.rnn.pad_sequence([i[:-1] for i in inputs],
            batch_first=True)[X_idx]
        Z = nn.utils.rnn.pack_padded_sequence(X, X_len - 1, batch_first=True)
        Xy = nn.utils.rnn.pad_sequence([i[1:] for i in inputs],
            batch_first=True)[X_idx]
        Zy = nn.utils.rnn.pack_padded_sequence(X, X_len - 1, batch_first=True)
        return Z, Zy.data

class Model(nn.Module):
    def __init__(self, input_size, hidden_size=256, z_dim=128):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(input_size, hidden_size)
        #Encoder.
        self.rnn_enc = nn.GRU(hidden_size, hidden_size, batch_first=True,
            bidirectional=True)
        self.lin_mu = nn.Linear(hidden_size * 2, z_dim)
        self.lin_ls = nn.Linear(hidden_size * 2, z_dim)
        #Sampler.
        self.lin_h = nn.Linear(z_dim, hidden_size)
        #Decoder.
        self.rnn_dec = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lin_o = nn.Linear(hidden_size, input_size)

    def _encode(self, x):
        _, h = self.rnn_enc(x)
        h = h.transpose(0, 1).reshape(-1, self.hidden_size * 2)
        return self.lin_mu(h), self.lin_ls(h)

    def _sample_z(self, mu, log_sigma):
        eps = torch.randn_like(log_sigma)
        std = torch.exp(log_sigma / 2)
        return self.lin_h(mu + std * eps)

    def _decode(self, x, h):
        out, _ = self.rnn_dec(x, h.unsqueeze(0))
        return F.log_softmax(self.lin_o(out.data), dim=1)

    def forward(self, x):
        assert type(x) is torch.nn.utils.rnn.PackedSequence
        x = torch.nn.utils.rnn.PackedSequence(
            data=self.emb(x.data), batch_sizes=x.batch_sizes)
        mu, log_sigma = self._encode(x)
        h = self._sample_z(mu, log_sigma)
        out = self._decode(x, h)
        return out, mu, log_sigma

    def encode(self, x):
        if type(x) is torch.nn.utils.rnn.PackedSequence:
            x = torch.nn.utils.rnn.PackedSequence(
                data=self.emb(x.data), batch_sizes=x.batch_sizes)
        elif type(x) == torch.Tensor:
            x = self.emb(x)
        else:
            raise ValueError('Unknown tensor type.')
        return self._encode(x)[0]

    def sample(self, h):
        return self.lin_h(h)

    def decode(self, x, h):
        x = self.emb(x)
        _, h = self.rnn_dec(x, h)
        return F.softmax(self.lin_o(h), dim=2), h

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = nn.NLLLoss(reduction='mean')

    def forward(self, x_pred, x, mu, log_sigma):
        rec_loss = self.loss(x_pred, x)
        div_loss = (log_sigma.exp() + mu.pow(2) - 1 - log_sigma).mean()
        return (rec_loss + div_loss) / 2, rec_loss, div_loss


#%% Init data.
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


#%% Init model.
model = Model(len(dataset.char_to_ix)).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = Loss()


#%% Train.
max_unimproved_epochs, unimproved_epochs = 15, 0
loss_min = float('inf')
for epoch in range(1000):
    start_time = time.time()
    #Training.
    model.train()
    losses = []
    for X, y in dataloader_train:
        X, y = X.to(device), y.to(device)
        X_pred, mu, log_sigma = model(X)
        loss, _, _ = loss_fn(X_pred, y, mu, log_sigma)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.detach())
    loss_train = torch.tensor(losses).mean().item()
    #Testing.
    model.eval()
    losses, rec_arr, div_arr = [], [], []
    with torch.no_grad():
        for X, y in dataloader_test:
            X, y = X.to(device), y.to(device)
            X_pred, mu, log_sigma = model(X)
            loss, rec_loss, div_loss = loss_fn(X_pred, y, mu, log_sigma)
            losses.append(loss)
            rec_arr.append(rec_loss)
            div_arr.append(div_loss)
    loss_test = torch.tensor(losses).mean().item()
    rec_test = torch.tensor(rec_arr).mean().item()
    div_test = torch.tensor(div_arr).mean().item()
    #Feedback.
    print(f'E{epoch} TRAIN:{loss_train:.3f} TEST:{loss_test:.3f}'
        f' REC:{rec_test:.3f} DIV:{div_test:.3f}'
        f' TOOK:{time.time() - start_time:.1f}s')
    #Save state & early stopping.
    unimproved_epochs += 1
    if loss_test < loss_min:
        torch.save(model.state_dict(), 'models/vae_name_gen.model')
        loss_min = loss_test
        unimproved_epochs = 0
    if unimproved_epochs > max_unimproved_epochs:
        print(f'E {epoch} Early stopping. BEST TEST: {loss_min:.3f}')
        break



#%% Quit here if ran as script.
if __name__ == '__main__':
    quit()


#%% Init model.
device = torch.device('cpu')
model = Model(len(dataset.char_to_ix)).to(device)
model.load_state_dict(
    torch.load('models/vae_name_gen.model', map_location=device))
model.eval()


#%% Reconstruct & interpolate.
def reconstruct(name, max_len=50):
    device = torch.device('cpu')
    inp = torch.tensor([dataset.char_to_ix[c] for c in CHAR_START + name],
        dtype=torch.int64, device=device, requires_grad=False).unsqueeze(0)
    with torch.no_grad():
        h = model.sample(model.encode(inp)).unsqueeze(0)
        out = ''
        x = torch.tensor([[dataset.char_to_ix[CHAR_START]]],
            dtype=torch.int64, device=device, requires_grad=False)
        while len(out) < max_len:
            x, h = model.decode(x, h)
            x = x.argmax(dim=2)
            out += dataset.ix_to_char[x.item()]
            if out[-1] == CHAR_END:
                return out[:-1]
        return out

def interpolate(name_a, name_b, n_items=10, max_len=50):
    device = torch.device('cpu')
    inp_a = torch.tensor([dataset.char_to_ix[c] for c in CHAR_START + name_a],
        dtype=torch.int64, device=device, requires_grad=False).unsqueeze(0)
    inp_b = torch.tensor([dataset.char_to_ix[c] for c in CHAR_START + name_b],
        dtype=torch.int64, device=device, requires_grad=False).unsqueeze(0)
    with torch.no_grad():
        h_a = model.encode(inp_a)
        h_b = model.encode(inp_b)
        results = []
        for mul in torch.linspace(0, 1, n_items):
            h = model.sample((h_a * (1 - mul) + h_b * mul)).unsqueeze(0)
            out = ''
            x = torch.tensor([[dataset.char_to_ix[CHAR_START]]],
                dtype=torch.int64, device=device, requires_grad=False)
            while len(out) < max_len:
                x, h = model.decode(x, h)
                x = x.argmax(dim=2)
                out += dataset.ix_to_char[x.item()]
                if out[-1] == CHAR_END:
                    out = out[:-1]
                    break
            results.append(out)
    return results

reconstruct('pavel')
interpolate('pavel', 'auttenberg')


#%% Show distribution.
mu_arr = []
with torch.no_grad():
    for X, _ in dataloader_train:
        X = X.to(device)
        mu_arr.append(model.encode(X))
    for X, _ in dataloader_test:
        X = X.to(device)
        mu_arr.append(model.encode(X))
mu_arr = torch.cat(mu_arr, dim=0).cpu().numpy()
mu_pca = PCA(n_components=2, whiten=True).fit_transform(mu_arr)
plt.scatter(mu_pca[:, 0], mu_pca[:, 1], s=0.1)
