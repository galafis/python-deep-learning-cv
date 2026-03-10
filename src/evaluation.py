"""
Model Evaluation Module
Confusion matrix and classification metrics.
"""

import numpy as np
from typing import Dict, List, Optional


class ModelEvaluator:
    """Evaluates classification models with confusion matrix and metrics."""

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                         num_classes: Optional[int] = None) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            y_true: Ground truth labels (1D array of class indices).
            y_pred: Predicted labels (1D array of class indices).
            num_classes: Number of classes.

        Returns:
            Confusion matrix of shape (num_classes, num_classes).
        """
        if num_classes is None:
            num_classes = max(int(y_true.max()), int(y_pred.max())) + 1

        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true.flatten(), y_pred.flatten()):
            cm[int(t), int(p)] += 1
        return cm

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy."""
        return float(np.mean(y_true.flatten() == y_pred.flatten()))

    def precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray,
                            num_classes: Optional[int] = None) -> Dict:
        """Compute per-class precision, recall, and F1 score."""
        cm = self.confusion_matrix(y_true, y_pred, num_classes)
        nc = cm.shape[0]

        metrics = {}
        for c in range(nc):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

            metrics[c] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        return metrics

    def classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None) -> str:
        """Generate a text classification report."""
        prf = self.precision_recall_f1(y_true, y_pred)
        acc = self.accuracy(y_true, y_pred)

        lines = [f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}"]
        lines.append("-" * 51)

        for c, m in prf.items():
            name = class_names[c] if class_names and c < len(class_names) else str(c)
            lines.append(f"{name:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")

        lines.append("-" * 51)
        lines.append(f"Accuracy: {acc:.4f}")
        return "\n".join(lines)
