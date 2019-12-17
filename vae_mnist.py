import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#%% Init data.
BATCH_SIZE = 124
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST('./data/mnist', train=True,
    download=True, transform=transforms)
test_dataset = torchvision.datasets.MNIST('./data/mnist', train=False,
    download=True, transform=transforms)
train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



#%%
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.lin_h = nn.Linear(input_dim, hidden_dim)
        self.lin_mean = nn.Linear(hidden_dim, z_dim)
        self.lin_log_sigma = nn.Linear(hidden_dim, z_dim)
    def forward(self, x):
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
        z_mean, z_log_sigma = self.encoder(x)
        x_pred = self.decoder(self.sample_z(z_mean, z_log_sigma))
        return x_pred, z_mean, z_log_sigma


#%%
input_dim = output_dim = 28 * 28
hidden_dim = 128
z_dim = 2

encoder = Encoder(input_dim, hidden_dim, z_dim)
decoder = Decoder(z_dim, hidden_dim, output_dim)
model = VAE(encoder, decoder).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

def vae_loss(x_pred, x, z_mean, z_log_sigma):
    reconstruction_loss = F.binary_cross_entropy(x_pred, x, reduction='sum')
    kl_loss = - 0.5 * torch.sum(1 + z_log_sigma - z_mean.pow(2) - z_log_sigma.exp())
    #kl_loss = 0.5 * torch.sum(z_log_sigma.exp() + z_mean.pow(2) - 1.0 - z_log_sigma)
    return reconstruction_loss + kl_loss



#%%
best_test_loss = float('inf')
for e in range(50):
    #Training.
    model.train()
    train_loss = 0
    for X, _ in train_iterator:
        X = X.reshape(-1, 28 * 28).to(device)
        optim.zero_grad()
        X_pred, z_mean, z_log_sigma = model(X)
        loss = vae_loss(X_pred, X, z_mean, z_log_sigma)
        loss.backward()
        train_loss += loss.item()
        optim.step()
    #Testing.
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, _ in test_iterator:
            X = X.reshape(-1, 28 * 28).to(device)
            X_pred, z_mean, z_log_sigma = model(X)
            loss = vae_loss(X_pred, X, z_mean, z_log_sigma)
            test_loss += loss.item()
    #Feedback.
    train_loss /= len(train_dataset)
    test_loss /= len(test_dataset)
    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')
    #Early stopping.
    if best_test_loss > test_loss:
        best_test_loss = test_loss
        patience_counter = 1
    else:
        patience_counter += 1
    if patience_counter > 3:
        break





#%% sample and generate a image
model.eval()
z = torch.randn(1, z_dim, device=device)
img = model.decoder(z).squeeze(0).reshape(28, 28)\
    .detach().cpu().numpy()
plt.figure()
plt.imshow(img, cmap='gray')
plt.show()



#%% Show all.
model.eval()
z_mean_arr, labels_arr = [], []
with torch.no_grad():
    for i, (X, label) in enumerate(test_iterator):
        X = X.reshape(-1, 28 * 28).to(device)
        _, z_mean, _ = model(X)
        z_mean_arr.append(z_mean)
        labels_arr.append(label)
z_mean = torch.cat(z_mean_arr, dim=0).detach().cpu()
labels = torch.cat(labels_arr, dim=0)
plt.figure(figsize=(10, 10))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
plt.colorbar()
plt.show()



#%% 2d manifold
model.eval()
digit_size, n, radius = 28, 15, 5
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
plt.imshow(figure.detach().numpy())
plt.show()
