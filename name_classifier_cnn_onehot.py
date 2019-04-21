import os
import unicodedata
import string
import torch
import torch.nn as nn
import torch.nn.functional as F



#GPU ready classifier with large batch using packed sequence.
#1D CNN on one-hot encoded char vectors.



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



#Make X, y
batch_size = len(names)
seq_len = max([len(name) for name in names])
input_size = len(char_to_ix)
output_size = len(cate_to_ix)

X = torch.zeros(batch_size, input_size, seq_len,
    dtype=torch.float32, device=device, requires_grad=False)
y = torch.tensor([cate_to_ix[category] for category in categories],
    dtype=torch.int64, device=device, requires_grad=False)
for n, name in enumerate(names):
    for c, char in enumerate(name):
        ix = char_to_ix[char]
        X[n][ix][c] = 1



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
class Conv1dClassifier(nn.Module):
    def __init__(self, seq_len, input_size, output_size):
        super(Conv1dClassifier, self).__init__()
        #6 convolutions. 2 of each.
        kernel_size = 5
        self.conv_a0 = torch.nn.Conv1d(input_size, 1, kernel_size)
        self.conv_a1 = torch.nn.Conv1d(input_size, 1, kernel_size)
        kernel_size = 4
        self.conv_b0 = torch.nn.Conv1d(input_size, 1, kernel_size)
        self.conv_b1 = torch.nn.Conv1d(input_size, 1, kernel_size)
        kernel_size = 3
        self.conv_c0 = torch.nn.Conv1d(input_size, 1, kernel_size)
        self.conv_c1 = torch.nn.Conv1d(input_size, 1, kernel_size)
        #Max value from all outputs.
        self.lin1 = nn.Linear(6, output_size)
    def forward(self, x):
        #batch, n_channels, seq_len, input_size = x.shape
        batch_size = x.shape[0]
        x = torch.cat((
            self.conv_a0(x).max(dim=2)[0].reshape(batch_size, -1),
            self.conv_a1(x).max(dim=2)[0].reshape(batch_size, -1),
            self.conv_b0(x).max(dim=2)[0].reshape(batch_size, -1),
            self.conv_b1(x).max(dim=2)[0].reshape(batch_size, -1),
            self.conv_c0(x).max(dim=2)[0].reshape(batch_size, -1),
            self.conv_c1(x).max(dim=2)[0].reshape(batch_size, -1)
        ), dim=1)
        x = self.lin1(F.relu(x))
        return F.log_softmax(x, dim=1)



#%% Training.
batch_size, input_size, seq_len = X.shape
model = Conv1dClassifier(seq_len, input_size, output_size)
if torch.cuda.is_available():
    model.cuda()
loss_fn = nn.NLLLoss(weight=weights, reduction='mean')
#optimizer = torch.optim.SGD(model.parameters(), lr=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for t in range(251):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    if t % 10 == 0:
        print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



#%% Predict.
ix_to_cate = {ix: cate for cate, ix in cate_to_ix.items()}
with torch.no_grad():
    X_t = torch.zeros(1, input_size, seq_len,
        dtype=torch.float32, device=device, requires_grad=False)
    for c, char in enumerate('Xiao'):
        ix = char_to_ix[char]
        X_t[0][ix][c] = 1
    y_pred = model(X_t).reshape(-1)
    values, indexes = torch.exp(y_pred).topk(5)
    for i in range(len(values)):
        value, index = values[i].item(), indexes[i].item()
        print(round(value, 2), '% -', ix_to_cate[index])
