import torch
from torch.utils.data import Dataset
from src.data.loader import JetDataset
from src.data.preprocess import image_to_graph

class GraphJetDataset(Dataset):
    def __init__(self, data_path, max_samples=5000):
        self.base_dataset = JetDataset(data_path, max_samples=max_samples)
        self.n = len(self.base_dataset)

    def __len__(self):
        return self.n

    def get_labels(self):
        return self.base_dataset.y

    def __getitem__(self, idx):
        # We need to return a Data object for torch_geometric
        image, label = self.base_dataset[idx]
        graph = image_to_graph(image.numpy(), label.item())

        if graph is None:
            # Simple fallback: try next sample
            return self.__getitem__((idx + 1) % self.n)

        return graph
