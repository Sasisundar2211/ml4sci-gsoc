from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def compute_classification_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "confusion_matrix": cm
    }
