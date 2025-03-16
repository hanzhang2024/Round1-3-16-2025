import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---- 1. Dummy Dataset ----
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=50):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.targets = torch.sum(self.data[:, ::2], dim=1)  # Sum of even indices

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# ---- 2. Learnable Positional Encoding ----
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.positions = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.02)  # Small initialization

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.positions[:seq_len, :].unsqueeze(0)  # Ensure correct slicing

# ---- 3. Model ----
class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = LearnablePositionalEncoding(max_seq_len, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=2, batch_first=True),
            num_layers=num_layers
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq, embed)
        x = self.pos_encoder(x)  # Add positional encodings
        x = self.transformer(x)  # Pass through transformer layers
        x = x.mean(dim=1)  # Pooling
        return self.head(x).squeeze()

# ---- 4. Training Loop ----
dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Model(vocab_size=50, embed_dim=16, max_seq_len=10)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Verify Positional Encodings Change
initial_positions = model.pos_encoder.positions.clone().detach()

for epoch in range(5):
    total_loss = 0
    for batch, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = loss_fn(outputs, targets.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# Check if positional encodings were updated
assert not torch.allclose(initial_positions, model.pos_encoder.positions.detach()), "Positional encodings did not update!"

# ---- 5. Visualization of Learned Positional Embeddings ----
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

positions = model.pos_encoder.positions.detach().numpy()
pca = PCA(n_components=2).fit_transform(positions)

plt.scatter(pca[:, 0], pca[:, 1], c=range(len(positions)), cmap='viridis')
plt.colorbar(label="Position Index")
plt.title("Learned Positional Encodings (PCA)")
plt.show()
