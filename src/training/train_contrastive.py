import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os

from src.data.loader import JetDataset
from src.data.augmentations import augment
from src.models.contrastive import ContrastiveModel

def info_nce_loss(z1, z2, temperature=0.2):
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    # For each row i, the positive is at index (i + batch_size) % (2 * batch_size)
    labels = (torch.arange(2 * batch_size) + batch_size) % (2 * batch_size)
    labels = labels.to(z.device)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -1e9)

    return F.cross_entropy(sim, labels)

def train(data_path, config_path="configs/contrastive.yaml"):
    # Load config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'data': {'batch_size': 64, 'max_samples': 20000},
            'training': {'epochs': 30, 'lr': 5e-4, 'temperature': 0.2}
        }

    print(f"Loading data from {data_path}...")
    dataset = JetDataset(data_path, max_samples=config['data']['max_samples'])
    loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ContrastiveModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    os.makedirs("outputs/models", exist_ok=True)

    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0

        for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
            x = x.to(device)

            x1 = augment(x)
            x2 = augment(x)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = info_nce_loss(z1, z2, temperature=config['training']['temperature'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "outputs/models/contrastive.pt")
    print("Training complete. Model saved to outputs/models/contrastive.pt")

if __name__ == "__main__":
    data_path = "dataset.hdf5"
    if not os.path.exists(data_path):
        data_path = "data/dataset.npz"
        
    train(data_path)
