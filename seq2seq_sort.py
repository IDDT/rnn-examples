import os
import unicodedata
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split



#%% Settings.
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 128



#%% Make data.
class NumbersDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.device = torch.device('cpu')
        self.data = []
        for i in range(50000):
            beg = torch.randint(90, (1,)).item()
            end = beg + torch.randint(low=3, high=10, size=(1,)).item()
            self.data.append(torch.arange(start=beg, end=end).tolist())
        ALL_NUMS = set([num for numbers in self.data for num in numbers])
        self.char_to_ix = {x: x for x in sorted(ALL_NUMS)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.make_input_vect(self.data[ix])

    def make_input_vect(self, doc:list):
        return torch.tensor(doc,
            dtype=torch.int64, device=self.device, requires_grad=False)

class RandomNumbersDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.device = torch.device('cpu')
        self.data = []
        for i in range(50000):
            indices = torch.arange(100)
            mask = ((torch.randn(100) - torch.tensor(-1.2)).clamp(min=0) == 0)
            numbers = indices[mask].tolist()
            if len(numbers) > 1:
                self.data.append(numbers)
        ALL_NUMS = set([num for numbers in self.data for num in numbers])
        self.char_to_ix = {x: x for x in sorted(ALL_NUMS)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.make_input_vect(self.data[ix])

    def make_input_vect(self, doc:list):
        return torch.tensor(doc,
            dtype=torch.int64, device=self.device, requires_grad=False)


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.device = torch.device('cpu')
        self.names = []
        for filename in os.listdir('data/names'):
            if filename.endswith('.txt'):
                with open('data/names/' + filename) as f:
                    for line in f:
                        self.names.append(self.to_ascii(line.strip().lower()))
        ALL_CHARS = set([char for name in self.names for char in name])
        self.char_to_ix = {x: i for i, x in enumerate(sorted(ALL_CHARS))}

    @staticmethod
    def to_ascii(s):
        out = []
        for char in unicodedata.normalize('NFD', s):
            if unicodedata.category(char) != 'Mn':
                out.append(char)
        return ''.join(out)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, ix):
        return self.make_input_vect(self.names[ix])

    def make_input_vect(self, doc:str):
        return torch.tensor([self.char_to_ix[char] for char in doc],
            dtype=torch.int64, device=self.device, requires_grad=False)

def make_x_y(inputs:list):
    #Create permuted inputs and indices to reconstruct them.
    p_inputs, r_indices = [], []
    for vect in inputs:
        permuted_indices = torch.randperm(len(vect))
        p_inputs.append(vect[permuted_indices])
        r_indices.append(permuted_indices.argsort())

    X_lengths, X_indices = torch.tensor([len(x) for x in p_inputs],
        dtype=torch.int16, device=torch.device('cpu'), requires_grad=False)\
        .sort(descending=True)
    X = nn.utils.rnn.pad_sequence(p_inputs, batch_first=True)[X_indices]
    Z = nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

    #Place targets according to inputs in Z.data
    y = torch.empty(len(Z.data), dtype=torch.int64,
        device=torch.device('cpu'), requires_grad=False)
    for i, index in enumerate(X_indices.tolist()):
        for t, target in enumerate(r_indices[index].tolist()):
            y[i + Z.batch_sizes[0:t].sum().item()] = target

    return Z, y



#%% Init data.
dataset = NumbersDataset()
n_test = len(dataset) // 20 // batch_size * batch_size
dataset_train, dataset_test = \
    random_split(dataset, (len(dataset) - n_test, n_test))
assert len(dataset_test) >= batch_size, "Batch size should be reduced."
dataloader_train = DataLoader(dataset_train, collate_fn=make_x_y,
    batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
dataloader_test = DataLoader(dataset_test, collate_fn=make_x_y,
    batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)



#%% Make model.
class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        hidden_size = 64
        self.emb = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNNCell(hidden_size, hidden_size)

    def _attn(self, x, x_all):
        '''
        Arguments:
            x:torch.Tensor (batch_size, feature_size)
                - Input (a) to calculate alignment scores.
            x_all:torch.Tensor (batch_size, seq_len, feature_size)
                - All inputs (b) to calculate alignment scores.
        Returns:
            log_weights:torch.Tensor (batch_size, seq_len)
                - Attention weights over the sequence length.
            output:torch.Tensor (batch_size, feature_size)
                - Weighted average of inputs (b).
        '''
        #Calculating alignment scores. (b×n×m) * (b×m×p) = (b×n×p)
        attn_weights = torch.bmm(x_all, x.unsqueeze(2)).squeeze(2)
        attn_weights = F.log_softmax(attn_weights, dim=1)
        #Get weighted average of inputs (b).
        output = (attn_weights.exp().unsqueeze(2) * x_all).sum(dim=1)
        #OR torch.bmm(attn_weights.exp().unsqueeze(1), x_pad).squeeze(1)
        return attn_weights, output

    def forward(self, x):
        assert type(x) is torch.nn.utils.rnn.PackedSequence
        #Convert indices to embeddings.
        x = nn.utils.rnn.PackedSequence(data=self.emb(x.data),
            batch_sizes=x.batch_sizes)
        #Get variables.
        seq_len, feature_size = x.batch_sizes.shape[0], x.data.shape[1]
        n_inputs, n_outputs = x.batch_sizes[0].item(), x.data.shape[0]
        device, dtype = x.data.device, x.data.dtype
        #Unpack inputs into zero padded matrix to calculate attention.
        x_pad, x_len = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        #Set context vector.
        context = x_pad.sum(dim=1) / x_len.unsqueeze(1).to(device)
        hidden = torch.zeros(n_inputs, feature_size, dtype=dtype,
            device=device, requires_grad=False)
        #Set placeholders for the output.
        y = torch.empty(n_outputs, seq_len, dtype=dtype,
            device=device, requires_grad=False)
        #Iterate through packed sequence manually.
        for i, batch_size in enumerate(x.batch_sizes.tolist()):
            #Slice variables - their size is only decreasing.
            context, hidden = context[0:batch_size], hidden[0:batch_size]
            x_pad = x_pad[0:batch_size]
            #Run through RNN.
            hidden = self.rnn(context, hidden)
            #Calculate alignment scores.
            attn_weights, context = self._attn(hidden, x_pad)
            #Get coordinates to set outputs & set outputs.
            beg_ix = x.batch_sizes[0:i].sum()
            end_ix = beg_ix + batch_size
            y[beg_ix:end_ix] = attn_weights
        return y

    def predict(self, x):
        assert type(x) is torch.Tensor
        #Apply embeddings.
        x = self.emb(x)
        #Get variables.
        n_inputs, seq_len, feature_size = x.shape[0], x.shape[1], x.shape[2]
        device, dtype = x.device, x.dtype
        #Set context vector.
        context = x.mean(dim=1)
        hidden = torch.zeros(n_inputs, feature_size, dtype=dtype,
            device=device, requires_grad=False)
        #Set output placeholder.
        y = torch.empty(n_inputs, seq_len, seq_len, dtype=dtype,
            device=device, requires_grad=False)
        #Iterate through items.
        for i in range(seq_len):
            #Slice variables - their size is only decreasing.
            hidden = self.rnn(context, hidden)
            #Calculate alignment scores.
            attn_weights, context = self._attn(hidden, x)
            #Set outputs.
            y[:,i] = attn_weights
        return y



#%% Init model.
model = Model(input_size=len(dataset.char_to_ix)).to(device)
loss_fn = nn.NLLLoss(reduction='mean')
optim = torch.optim.Adam(model.parameters(), lr=0.01)
#model.load_state_dict(torch.load('models/s2s_sort.model', map_location=device))



#%% Training.
max_unimproved_epochs, unimproved_epochs = 15, 0
loss_min = float('inf')
for epoch in range(1001):
    start_time = time.time()
    #Training.
    model.train()
    losses = []
    for Z, y in dataloader_train:
        Z, y = Z.to(device), y.to(device)
        y_pred = model(Z)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    loss_train = sum(losses) / len(losses)
    #Testing.
    model.eval()
    losses = []
    with torch.no_grad():
        for Z, y in dataloader_test:
            Z, y = Z.to(device), y.to(device)
            y_pred = model(Z)
            losses.append(loss_fn(y_pred, y).item())
    loss_test = sum(losses) / len(losses)
    #Feedback.
    print(f'E {epoch} TRAIN: {loss_train:.3f} TEST: {loss_test:.3f}'
        f' TOOK: {time.time() - start_time:.1f}s')
    #Save state & early stopping.
    unimproved_epochs += 1
    if loss_test < loss_min:
        torch.save(model.state_dict(), 'models/s2s_sort.model')
        loss_min = loss_test
        unimproved_epochs = 0
    if unimproved_epochs > max_unimproved_epochs:
        print(f'E {epoch} Early stopping. BEST TEST: {loss_min:.3f}')
        break



#%% Quit here if ran as script.
if __name__ == '__main__':
    quit()



#%% Sample.
model.eval()
char_to_ix = dataset.char_to_ix
ix_to_char = {ix:char for char, ix in char_to_ix.items()}
#char_indices = [char_to_ix[x] for x in 'david']
char_indices = [63, 64, 65, 66]

with torch.no_grad():
    x = torch.tensor(char_indices, dtype=torch.int64, device=device,
        requires_grad=False)[torch.randperm(len(char_indices))]
    y_pred = model.predict(x.unsqueeze(0)).squeeze(0)
    values, indices = y_pred.topk(1, dim=1)
    values, indices = values.squeeze(), indices.squeeze()

print([ix_to_char[i] for i in x.cpu().tolist()])
print([ix_to_char[i] for i in x[indices].cpu().tolist()])
print(values.exp())
