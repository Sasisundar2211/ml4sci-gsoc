import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.data.graph_loader import GraphJetDataset
from src.models.gnn import GNN
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import plot_roc_curve, plot_confusion_matrix
from tqdm import tqdm
import os
import yaml
from sklearn.model_selection import train_test_split

def train(data_path, config_path="configs/gnn.yaml"):
    # Load config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'data': {'batch_size': 32, 'k': 8},
            'model': {'hidden_dim': 128},
            'training': {'epochs': 20, 'lr': 1e-3}
        }

    print(f"Loading data from {data_path}...")
    dataset = GraphJetDataset(data_path, max_samples=5000)

    print("Preparing splits...")
    labels = dataset.get_labels()
    indices = list(range(len(dataset)))
    
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

    # Note: Accessing dataset[i] converts image to graph, which is expensive.
    # To avoid doing this multiple times, we can wrap it in a subset or just use the indices.
    # torch_geometric.loader.DataLoader handles the dataset nicely.
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = GNN(input_dim=4, hidden_dim=config['model']['hidden_dim']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_auc = 0.0
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                probs = torch.softmax(out, dim=1)[:, 1]
                all_labels.extend(batch.y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Compute metrics
        all_preds = [1 if p > 0.5 else 0 for p in all_probs]
        metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
        
        auc = metrics['roc_auc']
        scheduler.step(auc)
        
        print(f"Loss: {total_loss/len(train_loader):.4f} | Val ROC AUC: {auc:.4f} | Val Acc: {metrics['accuracy']:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "outputs/models/gnn.pt")
            print(f"Saved new best model with AUC: {auc:.4f}")
            
    # Final Visualizations
    model.load_state_dict(torch.load("outputs/models/gnn.pt"))
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    plot_roc_curve(all_labels, all_probs, model_name="GNN", save_path="outputs/plots/gnn_roc.png")
    plot_confusion_matrix(all_labels, all_preds, model_name="GNN", save_path="outputs/plots/gnn_cm.png")

    print("Training complete.")

if __name__ == '__main__':
    data_path = "dataset.hdf5"
    if not os.path.exists(data_path):
        data_path = "data/dataset.npz"
        
    train(data_path)
