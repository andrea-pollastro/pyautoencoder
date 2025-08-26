"""Example of training a VAE on MNIST dataset."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pyautoencoder import VAE, VAELoss

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create VAE model and loss function
latent_dim = 100
encoder = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 200),
    nn.ReLU(),
)

decoder = nn.Sequential(
    nn.Linear(latent_dim, 200),
    nn.ReLU(),
    nn.Linear(200, 784),
)

model = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
loss_fn = VAELoss(likelihood='bernoulli')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress tracking
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(100):
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
