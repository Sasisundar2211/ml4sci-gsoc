#!/bin/bash

# Set up environment
echo "Installing requirements..."
pip install -r requirements.txt

# Create output directories
mkdir -p outputs/models
mkdir -p outputs/plots

# Task 1: Autoencoder
echo "Running Autoencoder Training..."
python3 -m src.training.train_autoencoder

# Task 2: GNN
echo "Running GNN Training..."
python3 -m src.training.train_gnn

# Task 3: Contrastive Learning
echo "Running Contrastive Training..."
python3 -m src.training.train_contrastive

echo "Pipeline complete. Check outputs/ for results."
