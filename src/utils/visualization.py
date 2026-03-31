import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import os

def show_reconstructions(original, reconstructed, n=5, save_path=None):
    original = original.cpu().detach().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()

    for i in range(min(n, len(original))):
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        # Combine channels for visualization if it's 3D
        img_orig = np.mean(original[i], axis=0) if original[i].shape[0] == 3 else original[i][0]
        plt.imshow(img_orig, cmap="magma")
        plt.title(f"Original Sample {i+1}")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        img_recon = np.mean(reconstructed[i], axis=0) if reconstructed[i].shape[0] == 3 else reconstructed[i][0]
        plt.imshow(img_recon, cmap="magma")
        plt.title(f"Reconstructed Sample {i+1}")
        plt.colorbar()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(f"{save_path}_{i}.png")
        
        plt.tight_layout()
        if not save_path:
            plt.show()
        plt.close()

def plot_roc_curve(y_true, y_prob, model_name="Model", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    if not save_path:
        plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes=['Gluon', 'Quark'], model_name="Model", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    if not save_path:
        plt.show()
    plt.close()
