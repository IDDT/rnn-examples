import os
import unicodedata
import string
import torch
import torch.nn as nn
import torch.nn.functional as F



#GPU ready classifier with large batch using packed sequence.



ALL_CHARS = string.ascii_letters + " .,;'"
N_CHARS = len(ALL_CHARS)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



#Load data.
def to_ascii(s):
    out = []
    for char in unicodedata.normalize('NFD', s):
        if unicodedata.category(char) != 'Mn' and char in ALL_CHARS:
            out.append(char)
    return ''.join(out)

names, categories = [], []
for filename in os.listdir('data/names'):
    category = filename.strip('.txt').lower()
    if filename.endswith('.txt'):
        with open('data/names/' + filename) as f:
            for line in f:
                names.append(to_ascii(line.strip()))
                categories.append(category)

char_to_ix = {x: i for i, x in enumerate(ALL_CHARS)}
cate_to_ix = {x: i for i, x in enumerate(set(categories))}

def make_input_vect(name):
    vect = torch.zeros(len(name), len(char_to_ix), dtype=torch.float32,
        device=device, requires_grad=False)
    for c, char in enumerate(name):
        vect[c][char_to_ix[char]] = 1
    return vect



#Make X, y
batch_size = len(names)
input_size = len(char_to_ix)
output_size = len(cate_to_ix)

X, y = [], []
for i in range(len(names)):
    X.append(make_input_vect(names[i]))
    y.append(cate_to_ix[categories[i]])

X_lengths, X_indexes = torch.tensor([len(x) for x in X],
    dtype=torch.int16, device=device, requires_grad=False).sort(descending=True)
X = nn.utils.rnn.pad_sequence(X, batch_first=True)[X_indexes]
Z = nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
y = torch.tensor(y, dtype=torch.int64, device=device, requires_grad=False)[X_indexes]



#%% Make category weights.
counts = {}
for target in y:
    target = target.item()
    if target not in counts:
        counts[target] = 0
    counts[target] += 1
weights = torch.tensor([counts.get(i, 0) for i in range(len(cate_to_ix))],
    dtype=torch.float32, device=device, requires_grad=False)
min_weight = weights[weights > 0].min()
weights = min_weight / weights.clamp(min=min_weight)



#%% Model.
class RNNClassifier(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size,
            num_layers=1, nonlinearity='tanh', bias=True,
            batch_first=True, dropout=0, bidirectional=False)
        self.lin1 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        #batch, seq_len = x.shape[0], x.shape[1]
        h0 = torch.zeros(1, self.batch_size, self.hidden_size, dtype=torch.float32, device=device, requires_grad=True)
        output, h0 = self.rnn(x, h0)
        x = self.lin1(F.relu(h0.reshape(-1, self.hidden_size)))
        return F.log_softmax(x, dim=1)



#%% Training.
hidden_size = 128
model = RNNClassifier(batch_size, input_size, hidden_size, output_size)
if torch.cuda.is_available():
    model.cuda()
loss_fn = nn.NLLLoss(weight=weights, reduction='mean')
#optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for t in range(251):
    y_pred = model(Z)
    loss = loss_fn(y_pred, y)
    if t % 10 == 0:
        with torch.no_grad():
            print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



#%% Predict.
model.batch_size = 1
ix_to_cate = {ix: cate for cate, ix in cate_to_ix.items()}
with torch.no_grad():
    y_pred = model(make_input_vect('Xiao').reshape(1, -1, len(char_to_ix)))
    values, indexes = torch.exp(y_pred).topk(5)
    values, indexes = values.reshape(-1), indexes.reshape(-1)
    for i in range(len(values)):
        value, index = values[i].item(), indexes[i].item()
        print(round(value, 2), '% -', ix_to_cate[index])
