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
class BasicAttnClassifier(nn.Module):
    def __init__(self, input_size, seq_len, output_size):
        super(BasicAttnClassifier, self).__init__()
        char_emb_dim, pos_emb_dim = 50, 10
        emb_dim = char_emb_dim + pos_emb_dim
        self.char_emb = nn.Embedding(input_size, char_emb_dim,
            padding_idx=char_to_ix['<pad>'])
        self.pos_emb = nn.Embedding(seq_len, pos_emb_dim)


        self.conv_a = nn.Conv1d(emb_dim, 20, 7, padding=3)
        self.conv_b = nn.Conv1d(emb_dim, 20, 5, padding=2)
        self.conv_c = nn.Conv1d(emb_dim, 20, 3, padding=1)


        self.lin_o = nn.Linear(emb_dim * 2, output_size)
    def forward(self, x):
        #Convert list of itegers into dense char vectors.
        x = self.char_emb(x)
        #Create positional embeddings.
        xp = torch.arange(x.shape[1]).expand(x.shape[0], -1)
        xp = self.pos_emb(xp)
        #Concatenate.
        x = torch.cat((x, xp), dim=2)
        #Convert to keys, queries & values.
        x = x.transpose(1, 2)
        xx = torch.cat((
            self.conv_a(x).max(dim=2)[0],
            self.conv_b(x).max(dim=2)[0],
            self.conv_c(x).max(dim=2)[0]
        ), dim=1)
        attn_weights = torch.bmm(xx.unsqueeze(1), x).squeeze(1)
        attn_weights = F.softmax(attn_weights, dim=1)
        contexts = torch.bmm(x, attn_weights.unsqueeze(2)).squeeze(2)
        out = self.lin_o(torch.cat((contexts, xx), dim=1))
        return F.log_softmax(out, dim=1)



#%% Init model.
model = BasicAttnClassifier(input_size, seq_len, output_size).to(device)
loss_fn = nn.NLLLoss(weight=classweights, reduction='mean')
loss_fn_test = nn.NLLLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



#%% Training.
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
