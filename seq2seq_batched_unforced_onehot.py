import time

import re
import unicodedata
import string

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split



#Settings.
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 1024



#Preprocessing.
CHAR_BEG, CHAR_END = '<', '>'
ENG_CHARS = string.ascii_lowercase + string.digits
FRE_CHARS = string.ascii_lowercase + string.digits
SPECIAL_CHARS = "!#$%&'()*+,-./:;=?@^_~ "
ALL_CHARS_L1 = ''.join(sorted(set(FRE_CHARS + SPECIAL_CHARS)))
ALL_CHARS_L2 = ''.join(sorted(set(ENG_CHARS + SPECIAL_CHARS)))

REGEX_ESCAPE_CHARS = "!()*+-.=?"
ALL_CHARS_ESCAPED = ''.join([f'\\{x}' if x in REGEX_ESCAPE_CHARS else x \
    for x in ALL_CHARS_L1 + ALL_CHARS_L2])

def preprocess_fn(string):
    string = re.sub(r"[\t\n\r\x0b\x0c]", r" ", string.lower())
    string = re.sub(r'["`’‘]', r"'", string)
    string = re.sub(r"[\]\}>»]", r")", string)
    string = re.sub(r"[\[\{<«]", r"(", string)
    string = re.sub(r"[\|\\]", r"/", string)
    string = re.sub(r"[–‐]", r"-", string)
    string = re.sub(r"…", r"...", string)
    string = re.sub(r"‽", r"?", string)
    #replace non ASCII characters.
    string = re.sub(r"[àâа]", r"a", string)
    string = re.sub(r"æ", r"ae", string)
    string = re.sub(r"[çс]", r"c", string)
    string = re.sub(r"[éèêë]", r"e", string)
    string = re.sub(r"[îï]", r"i", string)
    string = re.sub(r"[ôöò]", r"o", string)
    string = re.sub(r"œ", r"oe", string)
    string = re.sub(r"[ùúûü]", r"u", string)
    string = re.sub(r"ÿ", r"y", string)
    #remove out of dictionary characters
    string = re.sub(f"[^{ALL_CHARS_ESCAPED}]", "", string)
    #remove double whitespace
    string = re.sub(r" {2,}", " ", string)
    return CHAR_BEG + string.strip() + CHAR_END

char_to_ix_l1 = {x: i for i, x in enumerate(CHAR_BEG + ALL_CHARS_L1 + CHAR_END)}
char_to_ix_l2 = {x: i for i, x in enumerate(CHAR_BEG + ALL_CHARS_L2 + CHAR_END)}



#Init dataset.
class Dataset(torch.utils.data.Dataset):
    def __init__(self, preprocess_fn):
        self.pairs = []
        with open('data/lang-pairs/eng-fra.txt', 'rt') as f:
            for line in f:
                eng, fra = line.strip().split('\t')
                eng, fra = preprocess_fn(eng), preprocess_fn(fra)
                self.pairs.append((fra, eng))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, ix):
        return self.pairs[ix]

def make_input_vect(string, char_to_ix, incl_last_char=True):
    if not incl_last_char:
        string = string[0:-1]
    indices = torch.tensor([char_to_ix[char] for char in string],
        dtype=torch.int64, device=torch.device('cpu'), requires_grad=False)
    return F.one_hot(indices, num_classes=len(char_to_ix)).type(torch.float32)

def make_x_y(inputs):
    l1, l2 = zip(*inputs)
    device = torch.device('cpu')

    #Make list of tensors.
    Xi = [make_input_vect(x, char_to_ix_l1, incl_last_char=True) for x in l1]
    X = [make_input_vect(x, char_to_ix_l2, incl_last_char=False) for x in l2]

    #Convert into unsorted packed sequences.
    Xi = nn.utils.rnn.pack_sequence(Xi, enforce_sorted=False)
    X = nn.utils.rnn.pack_sequence(X, enforce_sorted=False)

    #Place targets for NLLLoss according to inputs in X.data
    y = torch.zeros(len(X.data), dtype=torch.int64,
        device=device, requires_grad=False)
    for i, index in enumerate(X.sorted_indices):
        #Exclude the first character.
        for c, char in enumerate(l2[index.item()][1:]):
            y[i + X.batch_sizes[0:c].sum().item()] = char_to_ix_l2[char]
    return Xi, X, y


dataset = Dataset(preprocess_fn)
n_test = len(dataset) // 10
dataset_train, dataset_test = \
    random_split(dataset, (len(dataset) - n_test, n_test))
assert len(dataset_test) >= batch_size, "Batch size should be reduced."
dataloader_train = DataLoader(dataset_train, collate_fn=make_x_y,
    batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
dataloader_test = DataLoader(dataset_test, collate_fn=make_x_y,
    batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)



#Init model.
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        if type(x) == torch.nn.utils.rnn.PackedSequence:
            batch_size = x.batch_sizes[0].item()
            device, dtype = x.data.device, x.data.dtype
        elif type(x) == torch.Tensor:
            batch_size = x.shape[0]
            device, dtype = x.device, x.dtype
        else:
            raise ValueError('Unknown tensor type.')
        h0 = torch.zeros(1, batch_size, self.hidden_size,
            dtype=dtype, device=device)
        _, h = self.rnn(x, h0)
        return h

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRUCell(output_size, hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, forcing_ratio=0.5):
        assert type(x) == torch.nn.utils.rnn.PackedSequence
        assert type(h) == torch.Tensor
        assert h.shape[1] == x.batch_sizes[0]
        assert h.shape[2] == self.hidden_size
        #Arrange hidden according to packed sequence indexes.
        hidden = h.squeeze(0)[x.sorted_indices]
        #Set placeholders.
        inputs = torch.zeros_like(x.data[0:x.batch_sizes[0]])
        outputs = torch.empty_like(x.data)
        #Iterate through packed sequence manually.
        for i, batch_size in enumerate(x.batch_sizes.tolist()):
            #Get data for RNN step.
            beg_ix = x.batch_sizes[0:i].sum()
            end_ix = beg_ix + batch_size
            #Forcing ratio is 100% on first input.
            ratio = 1.0 if i == 0 else forcing_ratio
            #Replace random items from true data according to ratio.
            indices = torch.randperm(batch_size)[0:int(batch_size * ratio)]
            inputs[indices] = x.data[beg_ix:end_ix][indices]
            #Generate next hidden.
            hidden = self.rnn(inputs[0:batch_size], hidden[0:batch_size])
            #Generate predictions.
            outputs[beg_ix:end_ix] = F.log_softmax(self.lin(hidden), dim=1)
            #Genetate next inputs.
            topi = outputs[beg_ix:end_ix].topk(1)[1].flatten()
            inputs = F.one_hot(topi, num_classes=x.data.shape[1])\
                .type(torch.float32)
        return outputs



input_size = len(char_to_ix_l1)
output_size = len(char_to_ix_l2)
hidden_size = 100
encoder = Encoder(input_size, hidden_size).to(device)
decoder = Decoder(hidden_size, output_size).to(device)
loss_fn = nn.NLLLoss(reduction='mean')
optim_enc = torch.optim.Adam(encoder.parameters(), lr=0.001)
optim_dec = torch.optim.Adam(decoder.parameters(), lr=0.001)
# encoder.load_state_dict(torch.load('models/encoder_buo.model', map_location=device))
# decoder.load_state_dict(torch.load('models/decoder_buo.model', map_location=device))
# optim_enc.load_state_dict(torch.load('models/encoder_buo.optim', map_location=device))
# optim_dec.load_state_dict(torch.load('models/decoder_buo.optim', map_location=device))



#%% Training.
max_unimproved_epochs, unimproved_epochs = 15, 0
loss_min = float('inf')
for epoch in range(1001):
    start_time = time.time()
    #Training.
    encoder.train()
    decoder.train()
    losses = []
    for Xi, X, y in dataloader_train:
        #Workaround for bug in pytorch==1.2.
        #Xi, X, y = Xi.to(device), X.to(device), y.to(device)
        Xi, X, y = Xi.to(device=device), X.to(device=device), y.to(device)
        h = encoder(Xi)
        y_pred = decoder(X, h)
        loss = loss_fn(y_pred, y)
        optim_enc.zero_grad()
        optim_dec.zero_grad()
        loss.backward()
        optim_enc.step()
        optim_dec.step()
        losses.append(loss.item())
    loss_train = torch.tensor(losses, dtype=torch.float64).mean().item()
    #Testing.
    encoder.eval()
    decoder.eval()
    losses = []
    with torch.no_grad():
        for Xi, X, y in dataloader_test:
            #Workaround for bug in pytorch==1.2.
            #Xi, X, y = Xi.to(device), X.to(device), y.to(device)
            Xi, X, y = Xi.to(device=device), X.to(device=device), y.to(device)
            h = encoder(Xi)
            y_pred = decoder(X, h)
            losses.append(loss_fn(y_pred, y).item())
    loss_test = torch.tensor(losses, dtype=torch.float64).mean().item()
    #Feedback.
    print(f'E {epoch} TRAIN: {loss_train:.3f} TEST: {loss_test:.3f}'
        f' TOOK: {time.time() - start_time:.1f}s')
    #Save state & early stopping.
    unimproved_epochs += 1
    if loss_test < loss_min:
        torch.save(encoder.state_dict(), 'models/encoder_buo.model')
        torch.save(decoder.state_dict(), 'models/decoder_buo.model')
        torch.save(optim_enc.state_dict(), 'models/encoder_buo.optim')
        torch.save(optim_dec.state_dict(), 'models/decoder_buo.optim')
        loss_min = loss_test
        unimproved_epochs = 0
    if unimproved_epochs > max_unimproved_epochs:
        print(f'E {epoch} Early stopping. BEST TEST: {loss_min:.3f}')
        break



#Quit here if ran as script.
if __name__ == '__main__':
    quit()



#Sample.
encoder.eval()
decoder.eval()

ix_to_char_l2 = {value: key for key, value in char_to_ix_l2.items()}

def greedy_decode(hidden, max_length=100):
    out = '<'
    i = 0
    #Predicting.
    while len(out) < max_length:
        #Make char vector.
        char_ix = char_to_ix_l2[out[i]]
        char_vect = torch.zeros(1, 1, len(char_to_ix_l2),
            dtype=torch.float32, device=device, requires_grad=False)
        char_vect[0][0][char_ix] = 1
        #Run prediction.
        probas, hidden = decoder.predict(hidden, char_vect)
        #Add prediction if last char.
        if i == len(out) - 1:
            topv, topi = probas.topk(1)
            char = ix_to_char_l2[topi[0].item()]
            out += char
            if char == CHAR_END:
                break
        i += 1
    return out



l1, l2 = dataset[torch.randint(len(dataset), (1,)).item()]
print(l1, l2)
Xi = make_input_vect(l1, char_to_ix_l1, incl_last_char=True).unsqueeze(0).to(device)
h = encoder(Xi)
greedy_decode(h, max_length=100)
