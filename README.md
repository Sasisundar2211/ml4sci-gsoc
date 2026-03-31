# ML4SCI GSoC: Jet Classification using Autoencoder, GNN and Contrastive Learning

## Problem
Classify quark vs gluon jets using detector data from three channels:
* ECAL
* HCAL
* Tracks

Quark jets are narrow.
Gluon jets are wider and more diffuse.

Goal: learn representations that capture this structural difference.

## Approach
This work evaluates three approaches:
1. Autoencoder
2. Graph Neural Network
3. Contrastive Learning

Each method learns different aspects of jet structure.

### Results
GNN → Accuracy: 0.6930 | ROC AUC: 0.7833

## Evaluation

All models are evaluated using a strict **80/20 train-test split** with stratification to ensure fair performance measurement and generalization to unseen jet events.

## Task 1: Autoencoder
* CNN-based encoder-decoder
* Input: (3, 125, 125) jet images
* Loss: Mean Squared Error

Autoencoder reconstructs spatial energy distribution across ECAL, HCAL, and Tracks channels. It effectively captures the underlying detector topography across the three layers.

## Task 2: Graph Neural Network
### Pipeline
* Convert image → point cloud using non-zero pixels.
* Node features: (x, y, intensity, channel_id).
### Graph construction
* We construct graphs using **k-NN (k=8)** over spatial coordinates and define node features using position, intensity, and detector channel.
### Model
* 3rd-generation GraphSAGE with BatchNorm and Dropout.
* Binary classification for quark-gluon separation.

### Results
GNN → Accuracy: 0.6930 | ROC AUC: 0.7833

## Task 3: Contrastive Learning
### Method
* CNN Encoder + Deep Projection head
* InfoNCE loss (Temperature T=0.2)
* Data Augmentations: Random Crop, Gaussian Noise, Mirroring.
### Evaluation
* Linear/MLP Probe: Freeze encoder, train classifier on embeddings.
* Metric: Accuracy & ROC AUC on a held-out test set.

### Results
Contrastive → Accuracy: 0.7230 | ROC AUC: 0.7924

### Embedding Visualization
t-SNE visualization shows improved separation between quark and gluon jets using contrastive learning.

## Model Comparison

| Model | Test Accuracy | Test ROC AUC | Strength |
|------|----------|---------|----------|
| Autoencoder | - | - | Energy pattern reconstruction |
| GNN (baseline) | 0.6500 | 0.6800 | Initial graph baseline |
| GNN (improved) | 0.6930 | 0.7833 | Spatial particle interactions |
| **Contrastive** | **0.7230** | **0.7924** | Strong invariant features |

## Discussion

- **GNN performance improved** with feature normalization and architecture changes (specifically GraphSAGE with BatchNorm). The model effectively learns spatial correlations from the sparse point clouds.
- **Contrastive learning** provides better representation learning, improving separability between quark and gluon jets compared to the supervised GNN baseline.
- **Embedding Separation**: Visualization (t-SNE) confirms that contrastive features form more distinct clusters, indicating a more robust latent space for downstream classification tasks.

## Repository Structure
```
src/
  data/       # Optimized loaders and graph preprocessing
  models/     # Autoencoder, GNN (SAGE), and Contrastive models
  training/   # Config-driven training scripts
  utils/      # Metrics and visualization (ROC, CM)
notebooks/    # Result visualization and evaluation logic
outputs/      # Saved models and plots
```

## How to Run
```bash
# Set up environment
pip install -r requirements.txt

# Run full pipeline
./run.sh
```
