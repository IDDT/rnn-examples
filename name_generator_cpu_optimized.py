import os
import time
import unicodedata
import random
import string
import torch
import torch.nn as nn
import torch.nn.functional as F


#Feeding the whole word at once to rnn.
#Using optimizer instead of manually adjusting weights.


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
        dtype=torch.float32, device=device, requires_grad=False)
    arr_cate = torch.zeros(len(name) - 1, len(cate_to_ix),
        dtype=torch.float32, device=device, requires_grad=False)
    for c, char in enumerate(name[0:-1]):
        arr_name[c][char_to_ix[char]] = 1
        arr_cate[c][cate_to_ix[category]] = 1
    return torch.cat((arr_name, arr_cate), dim=1)

def make_target_vect(name):
    return torch.tensor([char_to_ix[x] for x in name[1:]],
        dtype=torch.int64, device=device, requires_grad=False)



#Make model.
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.o2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input):
        hidden = torch.zeros(1, 1, self.hidden_size, dtype=torch.float32, device=device)
        #Output of the rnn contains hidden layers of every step.
        output, _ = self.rnn(input, hidden)
        output = self.o2o(output)
        output = self.dropout(output)
        return self.log_softmax(output)

    def predict(self, input, hidden=None):
        #Predict char by char.
        if hidden is None:
            hidden = torch.zeros(1, 1, self.hidden_size, dtype=torch.float32, device=device)
        #Output and hidden are the same if seq_len = 1.
        _, hidden = self.rnn(input, hidden)
        output = self.o2o(hidden)
        output = self.dropout(output)
        return self.log_softmax(output), hidden

#Training.
def get_random_training_pair():
    i = random.randrange(len(names))
    return names[i], categories[i]


input_size = len(char_to_ix) + len(cate_to_ix)
output_size = len(char_to_ix)
hidden_size = 128
model = RNN(input_size, hidden_size, output_size)
if torch.cuda.is_available():
    model.cuda()
loss_fn = nn.NLLLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)


n_iters = 100000
losses = []
start_time = time.time()
for iter in range(1, n_iters + 1):

    name, category = get_random_training_pair()
    X = make_input_vect(name, category)
    y = make_target_vect(name)

    y_pred = model(X.reshape(1, X.shape[0], X.shape[1]))
    loss = loss_fn(y_pred.reshape(y_pred.shape[1], y_pred.shape[2]), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item() / X.shape[0])

    if iter % 5000 == 0:
        loss_reported = round(sum(losses[-50:]) / 50, 3)
        took_seconds = time.time() - start_time
        time_str = '{}m {}s'.format(int(took_seconds / 60), int(took_seconds % 60))
        print(iter, '{}%'.format(int(iter / n_iters * 100)), time_str, loss_reported)



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
