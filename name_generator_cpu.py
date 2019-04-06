import os
import time
import unicodedata
import random
import string
import torch
import torch.nn as nn
import torch.nn.functional as F


#Rewritten example from PyTorch tutorial with feeding the network char by char.


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
    #implementation using nn.RNN. takes ~20% longer to train.
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.o2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        #In for rnn with seq_len = 1, output and hidden are equal.
        _, hidden = self.rnn(input.reshape(1, 1, -1), hidden.reshape(1, 1, -1))
        output = self.o2o(hidden)
        output = self.dropout(output)
        output = self.softmax(output)
        return output.reshape(1, -1), hidden.reshape(1, -1)

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input.reshape(1, -1), hidden), dim=1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)



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
loss_fn = nn.NLLLoss()
learning_rate = 0.0005


start_time = time.time()
n_iters = 100000
losses = []
for iter in range(1, n_iters + 1):

    name, category = get_random_training_pair()
    X = make_input_vect(name, category)
    y = make_target_vect(name).reshape(-1, 1)

    hidden = model.init_hidden()
    model.zero_grad()
    total_loss = 0
    for i in range(X.shape[0]):
        output, hidden = model(X[i], hidden)
        loss = loss_fn(output, y[i])
        total_loss += loss

    total_loss.backward()
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    losses.append(total_loss.item() / X.shape[0])

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
        hidden = model.init_hidden()

        #The last letter will be used in prediction.
        for i in range(len(name) - 1):
            output, hidden = model(X[i], hidden)

        for i in range(max_length - len(name)):
            output, hidden = model(X[-1], hidden)
            topv, topi = output.topk(1)
            char = ix_to_char[topi[0][0].item()]
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
