import os
import time
import unicodedata
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



#%% Settings.
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



#%% Create charset and text preprocessing function.
#Make sure the padding key is at index 0.
ALL_CHARS = ['<pad>'] + list(string.ascii_letters) + list(" .,;'")
char_to_ix = {x: i for i, x in enumerate(ALL_CHARS)}

def normalize_string(string):
    out = []
    for char in unicodedata.normalize('NFD', string.strip()):
        if unicodedata.category(char) != 'Mn' and char in ALL_CHARS:
            out.append(char)
    return ''.join(out)



#%% Load data.
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
cate_to_ix = {x: i for i, x in enumerate(set(categories))}



#%% Additional variables.
seq_len = max([len(name) for name in names])
input_size = len(char_to_ix)
output_size = len(cate_to_ix)



#%% Split into train and test.
n_test = len(names) // 10
names, names_t = names[0:-n_test], names[-n_test:]
categories, categories_t = categories[0:-n_test], categories[-n_test:]



#%% Make X, y
def make_input_vectors(docs):
    out = torch.zeros(len(docs), seq_len, dtype=torch.int64,
        device=device, requires_grad=False)
    for d, doc in enumerate(docs):
        for c, char in enumerate(doc):
            out[d][c] = char_to_ix[char]
    return out

X = make_input_vectors(names)
y = torch.tensor([cate_to_ix[x] for x in categories],
    dtype=torch.int64, device=device, requires_grad=False)
X_t = make_input_vectors(names_t)
y_t = torch.tensor([cate_to_ix[x] for x in categories_t],
    dtype=torch.int64, device=device, requires_grad=False)



#%% Make classweights.
indices, counts = y.unique(return_counts=True)
indices, counts = indices.tolist(), counts.tolist()
classweights = torch.zeros(len(cate_to_ix), dtype=torch.float32,
    device=device, requires_grad=False)
for ix, count in zip(indices, counts):
    classweights[ix] = count
classweights = classweights.max() / classweights.clamp(min=1)



#%% Model.
class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads
        # These compute the queries, keys and values for all
        # heads (as a single concatenated vector)
        self.tokeys    = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues  = nn.Linear(k, k * heads, bias=False)
        # This unifies the outputs of the different heads into
        # a single k-vector
        self.unifyheads = nn.Linear(heads * k, k)
    def forward(self, x):
        b, t, k = x.shape
        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x)   .view(b, t, h, k)
        values  = self.tovalues(x) .view(b, t, h, k)
        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries / (k ** (1/4))
        keys    = keys / (k ** (1/4))
        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # - dot has size (b*h, t, t) containing raw weights
        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights
        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, k)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        return self.norm2(fedforward + x)

class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)
        # The sequence of transformer blocks that does all the
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)
        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.shape
        # generate position embeddings
        positions = torch.arange(t, device=x.device)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)
        x = tokens + positions
        x = self.tblocks(x)
        # Average-pool over the t dimension and project to class
        # probabilities
        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)



#%% Init model.
k = 60
heads = 2
depth = 2
seq_length = 22
num_tokens = len(char_to_ix)
num_classes = len(cate_to_ix)
model = Transformer(k, heads, depth, seq_length, num_tokens, num_classes).to(device)
loss_fn = nn.NLLLoss(weight=classweights, reduction='mean')
loss_fn_test = nn.NLLLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



#%% Training.
max_unimproved_epochs, unimproved_epochs = 100, 0
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
        torch.save(model.state_dict(), 'models/clf_tr.model')
    if unimproved_epochs > max_unimproved_epochs:
        minutes_took = (time.time() - start_time) / 60
        print(f'E {epoch} Early stopping. BEST TEST: {loss_min:.3f}')
        print(f'Took: {minutes_took:.1f}m')
        break



#%% Quit here if ran as script.
if __name__ == '__main__':
    quit()



#%% Predict.
ix_to_cate = {ix: cate for cate, ix in cate_to_ix.items()}
model.eval()


z = model.char_emb(x)
xp = torch.arange(z.shape[1]).expand(z.shape[0], -1)
xp = model.pos_emb(xp)
z = torch.cat((z, xp), dim=2)
raw_weights = torch.bmm(z, z.transpose(1, 2))
weights = F.softmax(raw_weights, dim=2)


weights
weights.mean(dim=1)


weights.mean(dim=1)
x = torch.bmm(weights, x)
#Mean of vectors to output.
x = F.relu(x.mean(dim=1))
x = self.o2o(x)

x

x = make_input_vectors(['Orlov'])
y_pred = model(x).reshape(-1)
values, indexes = torch.exp(y_pred).reshape(-1).topk(5)
for i in range(len(values)):
    value, index = values[i].item(), indexes[i].item()
    print(f'{value * 100:.1f}% - {ix_to_cate[index]}')
