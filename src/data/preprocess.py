import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

def image_to_graph(image, label, k=8):
    # image: (3, 125, 125)
    
    # Extract non-zero pixels
    # We find coordinates across all channels
    mask = (image.max(axis=0) > 0)
    coords = np.argwhere(mask) # (N, 2)
    
    if len(coords) == 0:
        return None
    
    points = []
    for x, y in coords:
        # Intensity across channels or just concat? 
        # Roadmap says [x, y, intensity, channel_id]
        # Since it's multichannel, we might need a node per pixel or a node with multiple intensities.
        # Physics images usually have intensities in different calorimeter layers.
        # But [x, y, intensity, channel_id] implies 4 features.
        # If we have 3 channels, we can take the sum or just use one.
        # Usually, we treat non-zero pixels from ANY channel as nodes.
        
        # Let's use the most prominent intensity and its channel id if we want exactly 4 features,
        # OR just use sum of intensity.
        # But standard for this task is often [x, y, intensity, layer_id].
        # Since we have 3 layers, we can have nodes for each layer OR nodes with 3 intensity features.
        # Roadmap says [x,y,intensity,channel_id] - this suggests a node per (pixel, channel).
        
        for c in range(image.shape[0]):
            intensity = image[c, x, y]
            if intensity > 0:
                points.append([x/125.0, y/125.0, intensity, float(c)])

    points = np.array(points) # (N, 4)
    if len(points) == 0:
        return None
        
    x_tensor = torch.tensor(points, dtype=torch.float)
    y_tensor = torch.tensor([label], dtype=torch.long)
    
    # k-NN on x, y coords
    nbrs = NearestNeighbors(n_neighbors=min(k, len(points))).fit(points[:, :2])
    dist, indices = nbrs.kneighbors(points[:, :2])
    
    edge_index = []
    edge_attr = []
    
    for i in range(len(points)):
        for j_idx, j in enumerate(indices[i]):
            if i == j: continue
            edge_index.append([i, j])
            edge_attr.append([dist[i, j_idx]])
            
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
