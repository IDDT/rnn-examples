import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt



#%% Init data.
batch_size = 128
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset_train = torchvision.datasets.MNIST('./data/mnist', train=True,
    download=True, transform=transforms)
dataset_test = torchvision.datasets.MNIST('./data/mnist', train=False,
    download=True, transform=transforms)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
    shuffle=True, drop_last=False, num_workers=0)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size,
    shuffle=False, drop_last=False, num_workers=0)



#%%
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.lin_h = nn.Linear(input_dim, hidden_dim)
        self.lin_mean = nn.Linear(hidden_dim, z_dim)
        self.lin_log_sigma = nn.Linear(hidden_dim, z_dim)
    def forward(self, x):
        #Encode input into z and z_distribution.
        h = F.relu(self.lin_h(x))
        z_mean = self.lin_mean(h)
        z_log_sigma = self.lin_log_sigma(h)
        return z_mean, z_log_sigma

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.lin_h = nn.Linear(z_dim, hidden_dim)
        self.lin_o = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        #Decode z into input.
        h = F.relu(self.lin_h(x))
        return torch.sigmoid(self.lin_o(h))

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    @staticmethod
    def sample_z(z_mean, z_log_sigma):
        epsilon = torch.randn_like(z_log_sigma)
        return z_mean + torch.exp(z_log_sigma / 2) * epsilon
    def forward(self, x):
        #Encode, sample from distribution, decode.
        z_mean, z_log_sigma = self.encoder(x)
        x_pred = self.decoder(self.sample_z(z_mean, z_log_sigma))
        return x_pred, z_mean, z_log_sigma

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x_pred, x, z_mean, z_log_sigma):
        r_loss = F.binary_cross_entropy(x_pred, x, reduction='none').sum(dim=1)
        kl_loss = 0.5 * (z_log_sigma.exp() + z_mean.pow(2) - 1 - z_log_sigma).sum(axis=1)
        return (r_loss + kl_loss).mean()



#%% Init model.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = output_dim = 28 * 28
hidden_dim, z_dim = 256, 2
loss_fn = VAELoss()
encoder = Encoder(input_dim, hidden_dim, z_dim)
decoder = Decoder(z_dim, hidden_dim, output_dim)
model = VAE(encoder, decoder).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
# model.load_state_dict(torch.load('models/vae_mnist.model', map_location=device))
# optim.load_state_dict(torch.load('models/vae_mnist.optim', map_location=device))



#%% Train.
max_unimproved_epochs, unimproved_epochs = 5, 0
loss_min = float('inf')
for epoch in range(1000):
    start_time = time.time()
    #Training.
    model.train()
    losses = []
    for X, _ in dataloader_train:
        X = X.reshape(-1, 28 * 28).to(device)
        optim.zero_grad()
        X_pred, z_mean, z_log_sigma = model(X)
        loss = loss_fn(X_pred, X, z_mean, z_log_sigma)
        loss.backward()
        losses.append(loss.item())
        optim.step()
    loss_train = sum(losses) / len(losses)
    #Testing.
    model.eval()
    losses = []
    with torch.no_grad():
        for X, _ in dataloader_test:
            X = X.reshape(-1, 28 * 28).to(device)
            X_pred, z_mean, z_log_sigma = model(X)
            loss = loss_fn(X_pred, X, z_mean, z_log_sigma)
            losses.append(loss.item())
    loss_test = sum(losses) / len(losses)
    #Feedback.
    print(f'E {epoch} TRAIN: {loss_train:.3f} TEST: {loss_test:.3f}'
        f' TOOK: {time.time() - start_time:.1f}s')
    #Save state & early stopping.
    unimproved_epochs += 1
    if loss_test < loss_min:
        torch.save(model.state_dict(), 'models/vae_mnist.model')
        torch.save(optim.state_dict(), 'models/vae_mnist.optim')
        loss_min = loss_test
        unimproved_epochs = 0
    if unimproved_epochs > max_unimproved_epochs:
        print(f'E {epoch} Early stopping. BEST TEST: {loss_min:.3f}')
        break



#%% Quit here if ran as script.
if __name__ == '__main__':
    quit()



#%% Sample and generate a image
model.eval()
indices = torch.randint(0, len(dataloader_test), (10,)).tolist()
X = [dataset_test[i][0] for i in indices]
X = torch.cat(X, dim=0).reshape(-1, input_dim).to(device)
with torch.no_grad():
    X_pred = model(X)[0]
X, X_pred = X.cpu(), X_pred.cpu()
f, axarr = plt.subplots(2, 10, figsize=(12, 2))
plt.xticks([], [])
for i in range(len(X)):
    axarr[0][i].imshow(X[i].reshape(28, 28), cmap='gray')
    axarr[1][i].imshow(X_pred[i].reshape(28, 28), cmap='gray')
    axarr[0][i].axis('off')
    axarr[1][i].axis('off')



#%% Show distribution.
model.eval()
z_mean_arr, labels_arr = [], []
with torch.no_grad():
    for i, (X, label) in enumerate(test_iterator):
        X = X.reshape(-1, 28 * 28).to(device)
        _, z_mean, _ = model(X)
        z_mean_arr.append(z_mean)
        labels_arr.append(label)
z_mean = torch.cat(z_mean_arr, dim=0).cpu()
labels = torch.cat(labels_arr, dim=0)
plt.figure(figsize=(10, 10))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
plt.colorbar()
plt.show()



#%% 2d manifold.
model.eval()
digit_size, n, radius = 28, 15, 2
figure = torch.empty((digit_size * n, digit_size * n))
grid_x = torch.linspace(-radius, radius, n)
grid_y = torch.linspace(-radius, radius, n)
for ix, x in enumerate(grid_x):
    for iy, y in enumerate(grid_y):
        z = torch.tensor([[x, y]]).to(device)
        digit = decoder(z).squeeze(0).reshape(digit_size, digit_size)
        x_low, x_high = ix * digit_size, (ix + 1) * digit_size
        y_low, y_high = iy * digit_size, (iy + 1) * digit_size
        figure[x_low:x_high, y_low:y_high] = digit
plt.figure(figsize=(10, 10))
plt.imshow(figure.detach().numpy(), cmap='gray')
plt.show()
