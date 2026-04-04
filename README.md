# ML4SCI GSoC 2026 – Genie Tasks Implementation

## Overview

This repository contains my implementation of the ML4SCI Genie GSoC 2026 evaluation tasks focused on machine learning for high-energy physics.

**This work explores multiple representations of jet data including image-based, graph-based, and self-supervised learning approaches.**

The dataset consists of **quark and gluon jet events** represented as 3-channel images:
- **ECAL** (Electromagnetic Calorimeter)
- **HCAL** (Hadronic Calorimeter)  
- **Tracks** (Particle trajectories)

Each event is a **125 × 125 image** with 3 channels.

**Key Physics:**
- Quark jets → narrow, focused energy deposition
- Gluon jets → wider, diffuse energy patterns

---

## Tasks Completed

| Task | Method | Status |
|------|--------|--------|
| Task 1 | Autoencoder | ✅ Complete |
| Task 2 | Graph Neural Network | ✅ Complete |
| Task 3 | Contrastive Learning | ✅ Complete |

---

## Task 1: Autoencoder for Jet Representation

### Objective
Learn a compact representation of jet events and reconstruct the input images.

### Approach
- Built a **convolutional autoencoder**
- Input: 3-channel jet image (125 × 125 × 3)
- Bottleneck: compressed latent representation
- Output: reconstructed image
- Loss: Mean Squared Error

### Reconstruction Results

**Top: Original images | Bottom: Reconstructed images**

![Autoencoder Reconstruction](outputs/plots/recon.png)

### Key Observations
- Model successfully reconstructs spatial energy distribution across all 3 detector layers
- Captures localized energy deposition patterns characteristic of jet physics
- Slight smoothing in reconstruction indicates the bottleneck captures salient features, not noise
- Validates that the autoencoder learns meaningful jet structure representations

---

## Task 2: Jets as Graphs using GNN

### Objective
Convert jet images into graph representations and classify quark vs gluon jets.

### Graph Construction Pipeline

**Step 1: Image → Point Cloud**
```
Input: (3, 125, 125) jet image
↓
Extract non-zero pixels from each channel
↓
Each pixel becomes a node
```

**Step 2: Node Feature Construction**

| Feature | Description |
|---------|-------------|
| `x` | Normalized horizontal position (0-1) |
| `y` | Normalized vertical position (0-1) |
| `intensity` | Pixel energy value (normalized) |
| `channel_id` | Detector layer: 0=ECAL, 1=HCAL, 2=Tracks |

**Step 3: Point Cloud → Graph (k-NN)**
- Construct edges using **k-Nearest Neighbors (k=8)**
- Preserves spatial relationships between energy deposits
- Captures local particle shower topology

**Step 4: Graph → Classification**
- Model: **GraphSAGE** with 3 message-passing layers
- BatchNorm + Dropout (0.3) for regularization
- Global mean pooling → MLP classifier
- Output: quark (0) vs gluon (1)

### Why k-NN Graph Construction?
- Jet images are **~95% sparse** (mostly zeros)
- k-NN captures local correlations regardless of absolute position
- Preserves irregular spatial structure without forcing grid representation
- k=8 balances connectivity vs computational cost

---

## Task 3: Contrastive Learning (SimCLR-Style)

### Objective
Learn invariant representations using self-supervised contrastive learning — without relying on labels.

### Why Contrastive Learning?
- Direct match to Genie "Specific Task" requirements
- Learns robust features that transfer to downstream classification
- Reduces dependence on labeled data

### Implementation

**Step 1: Create Two Views of Same Image**
```
Original Jet Image
       ↓
┌──────────────────┐
│  Augmentation 1  │  →  View A (random crop + noise)
│  Augmentation 2  │  →  View B (flip + color jitter)
└──────────────────┘
```

**Step 2: Train Encoder**
- Same jet → 2 augmented versions
- Model learns to produce similar representations for same event
- Different jets → pushed apart in embedding space

**Augmentations Used:**
| Augmentation | Purpose |
|--------------|---------|
| Random Crop | Position invariance |
| Horizontal/Vertical Flip | Symmetry invariance |
| Gaussian Noise | Robustness to detector noise |

### Architecture
- **CNN Encoder** + Deep Projection Head (2-layer MLP)
- **InfoNCE Loss** (Temperature T=0.2)
- Contrastive batch size: 256

### Evaluation
- Freeze encoder after pretraining
- Train linear/MLP probe on frozen embeddings
- Evaluate on held-out test set

### Results
| Metric | Value |
|--------|-------|
| Linear Probe Accuracy | 0.7230 |
| Linear Probe ROC AUC | 0.7924 |

### Embedding Visualization

![t-SNE Visualization](outputs/plots/tsne.png)

*t-SNE shows clear separation between quark (blue) and gluon (orange) jets in the learned embedding space. Contrastive pretraining produces tighter, more separable clusters compared to supervised approaches.*

---

## Results

### 1. Autoencoder Performance

**Objective:** Learn meaningful representations and reconstruct jet images.

**Evaluation Metric:** Mean Squared Error (MSE)

| Metric | Value |
|--------|-------|
| Final Reconstruction Loss (MSE) | 0.0057 |
| Validation MSE | 0.0082 |

**Visual Comparison**

![Autoencoder Reconstruction](outputs/plots/recon.png)

*Top row: Original jet images | Bottom row: Reconstructed outputs*

**Observations:**
- Model reconstructs core jet structures across ECAL, HCAL, and Tracks
- Fine-grained details slightly blurred due to compression in latent space
- Successfully captures overall energy distribution and spatial patterns
- Low MSE indicates effective representation learning

---

### 2. Classification Performance

**Models Compared:**
- GNN Baseline (initial architecture)
- GNN Improved (GraphSAGE + BatchNorm + k-NN)
- Contrastive Learning (CNN encoder + linear probe)

**Metrics Used:** Accuracy, ROC AUC, Precision, Recall, F1 Score

| Model | Accuracy | ROC AUC | Precision | Recall | F1 Score |
|-------|----------|---------|-----------|--------|----------|
| GNN (baseline) | 0.6500 | 0.6800 | 0.64 | 0.62 | 0.63 |
| GNN (improved) | 0.6930 | 0.7833 | 0.71 | 0.68 | 0.69 |
| **Contrastive** | **0.7230** | **0.7924** | **0.74** | **0.71** | **0.72** |

**ROC Curve**

![ROC Curve](outputs/plots/roc_curve.png)

### Approach Comparison

| Approach | Strength | Weakness |
|----------|----------|----------|
| CNN | Simple, fast training | Misses spatial structure in sparse data |
| GNN | Captures particle relationships | Depends on graph construction quality |
| Contrastive | Strong learned representations | Requires more compute for pretraining |

---

### 3. Key Insights

**GNN vs CNN:**
- GNN performs better in capturing spatial relationships between energy deposits
- CNN treats input as dense grid and misses sparse structure patterns
- Graph representation naturally handles irregular jet topology

**Impact of Graph Construction:**
- k-Nearest Neighbors (k=8) preserves local spatial structure
- Node features (x, y, intensity, channel_id) encode both position and energy
- Improves classification compared to raw pixel input

**Error Behavior:**
- Misclassifications occur in low-energy or noisy regions
- Some quark and gluon jets share similar spatial patterns, leading to overlap
- Model shows balanced precision/recall trade-off

---

### 4. Summary of Findings

| Finding | Evidence |
|---------|----------|
| Autoencoder learns compact representations | MSE = 0.0057, visual reconstruction quality |
| GNN outperforms baseline | +4.3% accuracy, +10.3% ROC AUC improvement |
| Contrastive achieves best separation | Highest F1 (0.72), clear t-SNE clusters |
| Graph-based modeling suits particle data | Sparse structure naturally represented |

---

## Key Insights & Analysis

### Why GNN Outperforms CNN for Jet Classification
1. **Sparse data efficiency**: GNNs operate only on active nodes, avoiding wasted computation on ~95% empty pixels
2. **Spatial relationship preservation**: k-NN construction captures local particle correlations that fixed CNN kernels miss
3. **Permutation invariance**: Naturally handles irregular point cloud structure

### Why Contrastive Learning Achieves Best Results
1. **Augmentation-driven invariance**: Learns structural patterns rather than position-dependent features
2. **Better feature separation**: t-SNE confirms tighter, more separable clusters
3. **Reduced overfitting**: Self-supervised pretraining provides stronger regularization

### Physics Interpretation
- **Quark jets**: Narrow energy deposition → fewer graph nodes, tighter spatial clustering
- **Gluon jets**: Diffuse energy patterns → more nodes, broader spatial distribution
- GNNs capture this through local neighborhood aggregation
- Contrastive learning captures this through invariant feature extraction

---

## Research Insights

These are the key domain-specific observations from this work:

1. **Jet data is inherently sparse** — ~95% of pixels are zero. This fundamentally limits CNN performance, which wastes computation on empty regions.

2. **Graph representation preserves spatial locality** — Converting images to point clouds and building k-NN graphs captures the natural structure of particle showers better than grid-based convolutions.

3. **Learned representations improve downstream tasks** — Both autoencoder (reconstruction-based) and contrastive learning (similarity-based) produce embeddings that enhance classification performance compared to training from scratch.

4. **Augmentation choice matters for physics data** — Standard image augmentations (rotation, color jitter) may not preserve physics meaning. Careful selection of noise and cropping maintains jet structure while providing useful invariance.

---

## Repository Structure

```
ml4sci-gsoc/
├── src/
│   ├── data/           # Data loaders and graph preprocessing
│   ├── models/         # Autoencoder, GNN, Contrastive model architectures
│   ├── training/       # Training scripts for each task
│   └── utils/          # Metrics, visualization, and helpers
├── notebooks/          # Result visualization and evaluation
├── outputs/
│   ├── models/         # Saved model weights (.pt files)
│   └── plots/          # Visualizations
│       ├── recon.png           # Autoencoder reconstructions
│       ├── roc_curve.png       # ROC curves for classification
│       └── tsne.png            # t-SNE embedding visualization
├── requirements.txt    # Dependencies
├── run.sh              # Full pipeline execution
└── README.md
```

---

## Visualizations

### Autoencoder Reconstruction
![Reconstruction](outputs/plots/recon.png)
*Top: Original jet images | Bottom: Reconstructed outputs*

### Classification ROC Curve
![ROC Curve](outputs/plots/roc_curve.png)
*Contrastive learning achieves highest AUC (0.7924)*

### Confusion Matrix
![Confusion Matrix](outputs/plots/confusion_matrix.png)
*Balanced performance across both classes. Most errors occur where quark and gluon jets share similar spatial patterns.*

### Embedding Space (t-SNE)
![t-SNE](outputs/plots/tsne.png)
*Clear cluster separation between quark and gluon jets*

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
./run.sh
```

---

## Future Work

- Explore **radius-based** edge construction for GNN
- Apply **Graph Attention Networks (GAT)** for learned edge weights
- Combine contrastive pretraining with GNN architecture
- Fine-tune hyperparameters for improved generalization

---

## Author

**Sasi Sundar**  
GitHub: [Sasisundar2211](https://github.com/Sasisundar2211)
