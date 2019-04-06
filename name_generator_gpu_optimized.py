import os
import time
import unicodedata
import string
import torch
import torch.nn as nn
import torch.nn.functional as F



CHAR_START, CHAR_END = '<', '>'
ALL_CHARS = CHAR_START + string.ascii_letters + " .,;'-" + CHAR_END
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
                names.append(CHAR_START + to_ascii(line.strip()) + CHAR_END)
                categories.append(category)

char_to_ix = {x: i for i, x in enumerate(ALL_CHARS)}
cate_to_ix = {x: i for i, x in enumerate(set(categories))}

def make_input_vect(name, category):
    arr_name = torch.zeros(len(name) - 1, len(char_to_ix),
        dtype=torch.float32, device=device)
    arr_cate = torch.zeros(len(name) - 1, len(cate_to_ix),
        dtype=torch.float32, device=device)
    for c, char in enumerate(name[0:-1]):
        arr_name[c][char_to_ix[char]] = 1
        arr_cate[c][cate_to_ix[category]] = 1
    return torch.cat((arr_name, arr_cate), dim=1)


batch_size = len(names)
X = [make_input_vect(names[i], categories[i]) for i in range(batch_size)]
X_lengths, X_indexes = torch.tensor([len(x) for x in X],
    dtype=torch.int16, device=device, requires_grad=False).sort(descending=True)
X = nn.utils.rnn.pad_sequence(X, batch_first=True)[X_indexes]
Z = nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)


#Place targets according to inputs in Z.data
y = torch.zeros(len(Z.data), dtype=torch.int64, device=device, requires_grad=False)
for i, index in enumerate(X_indexes):
    index = index.item()
    #Not taking the first letter of the name for targets.
    targets = [char_to_ix[char] for char in names[index][1:]]
    for t, target in enumerate(targets):
        y[i + Z.batch_sizes[0:t].sum().item()] = target



#Make model.
class RNN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.o2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size, dtype=torch.float32, device=device)
        #Output of the rnn contains hidden layers of every step.
        output, _ = self.rnn(input, hidden)
        #Only PackedSequence or batch_size == 1.
        if type(output) is not torch.nn.utils.rnn.PackedSequence:
            assert output.shape[0] == 1
        out = self.o2o(output[0])
        out = self.dropout(out)
        return F.log_softmax(out, dim=1)

    def predict(self, input, hidden=None):
        #Predict char by char. Batch size = 1.
        if hidden is None:
            hidden = torch.zeros(1, 1, self.hidden_size, dtype=torch.float32, device=device)
        #Output and hidden are the same if seq_len = 1.
        _, hidden = self.rnn(input, hidden)
        out = self.o2o(hidden)
        out = self.dropout(out)
        return F.log_softmax(out, dim=2), hidden



#Training.
batch_size = len(names)
input_size = len(char_to_ix) + len(cate_to_ix)
output_size = len(char_to_ix)
hidden_size = 128
model = RNN(batch_size, input_size, hidden_size, output_size)
if torch.cuda.is_available():
    model.cuda()
loss_fn = nn.NLLLoss(reduction='mean')
#optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
start_time = time.time()
for iter in range(301):

    y_pred = model(Z)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if iter % 10 == 0:
        took_seconds = time.time() - start_time
        time_str = '{}m {}s'.format(int(took_seconds / 60), int(took_seconds % 60))
        print(iter, time_str, losses[-1])



#Testing.
ix_to_char = {value: key for key, value in char_to_ix.items()}

def make_input_vect_test(name, category):
    arr_name = torch.zeros(len(name), len(char_to_ix),
        dtype=torch.float32, device=device)
    arr_cate = torch.zeros(len(name), len(cate_to_ix),
        dtype=torch.float32, device=device)
    for c, char in enumerate(name):
        arr_name[c][char_to_ix[char]] = 1
        arr_cate[c][cate_to_ix[category]] = 1
    return torch.cat((arr_name, arr_cate), dim=1)

def sample(category, start_str='', max_length=20):
    with torch.no_grad():
        name = CHAR_START + start_str
        X = make_input_vect_test(name, category)
        hidden = None
        for i in range(len(name) - 1):
            output, hidden = model.predict(X[i].reshape(1, 1, -1), hidden)
        for i in range(max_length - len(name)):
            output, hidden = model.predict(X[-1].reshape(1, 1, -1), hidden)
            topv, topi = output.reshape(-1).topk(1)
            char = ix_to_char[topi[0].item()]
            if char == CHAR_END:
                break
            name += char
            X = make_input_vect_test(name, category)
        return name[1:]

def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


samples('russian', 'RUS')

samples('german', 'GER')

samples('spanish', 'SPA')

samples('chinese', 'CHI')
