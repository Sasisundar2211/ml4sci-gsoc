import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loader import JetDataset
from src.models.autoencoder import AutoEncoder

def train(data_path, epochs=10, batch_size=32):

    dataset = JetDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for x, _ in tqdm(loader):
            x = x.to(device)

            recon = model(x)
            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

    torch.save(model.state_dict(), "outputs/models/autoencoder.pt")


if __name__ == "__main__":
    train("data/dataset.npz")
