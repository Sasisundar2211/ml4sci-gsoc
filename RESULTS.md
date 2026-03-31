# Results Summary

Autoencoder → Successfully reconstructs spatial energy distribution.
GNN → Accuracy: 0.6930 | ROC AUC: 0.7833 (Upgraded with k-NN, k=8, and BatchNorm)
Contrastive → Accuracy: 0.8100 | ROC AUC: 0.8350 (Re-training in progress for 0.85+)

**Observation:**
- The improved GNN captures spatial relationships much more effectively by using standardized k-NN graph construction (k=8) and specific node features [x, y, intensity, channel_id].
- Contrastive learning continues to provide the highest separability for quark vs gluon classification.
- Autoencoder provides a useful baseline for dimensionality reduction and energy pattern recovery.
