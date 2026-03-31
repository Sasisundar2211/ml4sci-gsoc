import numpy as np
import torch
from torch.utils.data import Dataset

class JetDataset(Dataset):
    def __init__(self, data_path, max_samples=None):
        if data_path.endswith('.hdf5') or data_path.endswith('.h5'):
            import h5py
            f = h5py.File(data_path, 'r')
            n = f['X_jets'].shape[0]
            if max_samples and max_samples < n:
                n = max_samples
            X = f['X_jets'][:n].astype(np.float32)
            y = f['y'][:n].astype(np.int64)
            f.close()
            # channels-last to channels-first: (N,125,125,3) -> (N,3,125,125)
            X = np.transpose(X, (0, 3, 1, 2))
        else:
            data = np.load(data_path, allow_pickle=True)
            if "X" in data:
                X = data["X"]
                y = data["y"]
            elif "images" in data:
                X = data["images"]
                y = data["labels"]
            else:
                raise ValueError("Unknown dataset format")
            X = X.astype(np.float32)
            y = y.astype(np.int64)
            if max_samples and max_samples < len(X):
                X = X[:max_samples]
                y = y[:max_samples]

        self.X = X
        self.y = y

        # normalize per channel
        for c in range(3):
            mean = self.X[:, c].mean()
            std = self.X[:, c].std() + 1e-6
            self.X[:, c] = (self.X[:, c] - mean) / std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
