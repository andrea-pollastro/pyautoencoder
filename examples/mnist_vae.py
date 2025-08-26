"""Example of training a VAE on MNIST dataset."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pyautoencoder import VAE, VAELoss

# Hyperparameters
latent_dim = 32
hidden_dim = 512
batch_size = 128
learning_rate = 1e-3
num_epochs = 50
beta = 1.0  # for β-VAE, increase for stronger disentanglement

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define encoder and decoder networks
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class Decoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=512, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # For MNIST pixel values in [0,1]
        )
    
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

# Create VAE model and loss function
encoder = Encoder(input_dim=784, hidden_dim=hidden_dim, output_dim=latent_dim)
decoder = Decoder(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=784)
model = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
loss_fn = VAELoss(beta=beta, likelihood='bernoulli')  # Binary images
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with progress tracking
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x)
        loss_info = loss_fn(output, x)
        
        # Compute gradients and update parameters
        loss_info.total.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss_info.total.item()
        total_recon += loss_info.components['log_likelihood'].item()
        total_kl += loss_info.components['kl_divergence'].item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'Loss: {loss_info.total.item():.4f}, '
                  f'Recon: {loss_info.components["log_likelihood"].item():.4f}, '
                  f'KL: {loss_info.components["kl_divergence"].item():.4f}')
    
    # Print epoch statistics
    avg_loss = total_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_kl = total_kl / len(train_loader)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} '
          f'(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})')
