import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================================
# 1. Dataset
# ============================================================================
class IdentityDataset(Dataset):
    def __init__(self, data_path):
        """
        Load data from .npy file
        Expected shape: (40, 900, 64)
        """
        data = np.load(data_path)  # Shape: (40, 900, 64)
        print(f"Loaded data shape: {data.shape}")
        
        # Reshape to (num_samples, embed_dim)
        # We'll treat each of the 900 patches from each of 40 images as a sample
        self.data = torch.FloatTensor(data).reshape(-1, data.shape[-1])  # (40*900, 64)
        print(f"Reshaped to: {self.data.shape}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Input and target are the same (identity task)
        x = self.data[idx]
        return x, x

# ============================================================================
# 2. Model
# ============================================================================
class SimpleNN(nn.Module):
    def __init__(self, embed_dim, use_relu=False, init_identity=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU() if use_relu else nn.Identity(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        if init_identity:
            # Initialize to identity mapping
            if use_relu:
                print("Warning: ReLU is enabled, perfect identity is impossible!")
            
            nn.init.eye_(self.net[0].weight)
            nn.init.zeros_(self.net[0].bias)
            nn.init.eye_(self.net[2].weight)
            nn.init.zeros_(self.net[2].bias)
            print("Initialized to identity mapping")
    
    def forward(self, x):
        return self.net(x)

# ============================================================================
# 3. Training Function
# ============================================================================
def train_identity_net(
    data_path='exp_features.npy',
    embed_dim=64,
    batch_size=128,
    epochs=10,
    lr=0.01,
    weight_decay=0.0,
    use_relu=False,
    init_identity=True
):
    """
    Train a simple NN to learn identity mapping
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    dataset = IdentityDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = SimpleNN(embed_dim, use_relu=use_relu, init_identity=init_identity).to(device)
    
    # Test initial loss
    with torch.no_grad():
        test_input = dataset.data[:100].to(device)
        test_output = model(test_input)
        initial_loss = nn.functional.mse_loss(test_output, test_input)
        print(f"Initial loss: {initial_loss.item():.6f}\n")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    print(f"Training with lr={lr}, weight_decay={weight_decay}, use_relu={use_relu}")
    print("="*70)
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    
    print("="*70)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_inputs = dataset.data.to(device)
        all_outputs = model(all_inputs)
        final_loss = criterion(all_outputs, all_inputs)
        
        # Compute error statistics
        errors = (all_outputs - all_inputs).abs()
        max_error = errors.max().item()
        mean_error = errors.mean().item()
        
        print(f"\nFinal Results:")
        print(f"  MSE Loss: {final_loss.item():.8f}")
        print(f"  Mean Absolute Error: {mean_error:.8f}")
        print(f"  Max Absolute Error: {max_error:.8f}")
    
    return model, losses

# ============================================================================
# 4. Comparison Function
# ============================================================================
def compare_configurations(data_path='exp_features.npy'):
    """
    Compare different configurations
    """
    configs = [
        {"name": "Identity Init, No Weight Decay", "weight_decay": 0.0, "init_identity": True, "use_relu": False},
        {"name": "Identity Init, With Weight Decay", "weight_decay": 0.05, "init_identity": True, "use_relu": False},
        {"name": "Random Init, No Weight Decay", "weight_decay": 0.0, "init_identity": False, "use_relu": False},
        {"name": "Identity Init + ReLU", "weight_decay": 0.0, "init_identity": True, "use_relu": True},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"{'='*70}")
        
        model, losses = train_identity_net(
            data_path=data_path,
            weight_decay=config['weight_decay'],
            init_identity=config['init_identity'],
            use_relu=config['use_relu'],
            epochs=5
        )
        
        results[config['name']] = losses
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, marker='o', label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Identity Learning: Different Configurations')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('identity_learning_comparison.png', dpi=150)
    print(f"\nPlot saved to: identity_learning_comparison.png")
    plt.show()

# ============================================================================
# 5. Main
# ============================================================================
if __name__ == "__main__":
    # Test single configuration
    print("Testing Identity Learning\n")
    if 0:
        model, losses = train_identity_net(
            data_path='exp_features.npy',
            weight_decay=0.0,  # Try 0.0 vs 0.05
            init_identity=True,
            use_relu=False,
            epochs=5
        )
    
    # Uncomment to compare multiple configurations
    compare_configurations('exp_features.npy')