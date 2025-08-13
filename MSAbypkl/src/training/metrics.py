import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score,classification_report, precision_score, recall_score

from scipy.stats import pearsonr
import logging
import csv

logger = logging.getLogger(__name__)

def calc_mae(preds, labels):
    """
    Calculate Mean Absolute Error (MAE) between predictions and labels.
    
    Args:
        preds (np.ndarray): Predicted values.
        labels (np.ndarray): Ground truth values.
    
    Returns:
        float: Mean absolute error.
    """
    # 添加 NaN 处理
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    
    # 找出有效的索引（非 NaN 且非 Inf）
    valid_mask = np.isfinite(preds) & np.isfinite(labels)
    
    if not np.any(valid_mask):
        return float('nan')
    
    return mean_absolute_error(labels[valid_mask], preds[valid_mask])

def calc_correlation(preds, labels):
    """
    Calculate Pearson correlation coefficient between predictions and labels.
    
    Args:
        preds (np.ndarray): Predicted values.
        labels (np.ndarray): Ground truth values.
    
    Returns:
        float: Pearson correlation value.
    """
    if np.std(preds) == 0 or np.std(labels) == 0:
        return 0.0
    
    return np.corrcoef(preds, labels)[0, 1]

def calc_binary_accuracy(preds, labels, threshold=0):
    """
    Calculate binary classification accuracy based on a threshold.
    
    Args:
        preds (np.ndarray): Predicted values.
        labels (np.ndarray): Ground truth values.
        threshold (float): Threshold to binarize predictions and labels.
    
    Returns:
        float: Binary classification accuracy.
    """
    binary_preds = (preds > threshold).astype(int)
    binary_labels = (labels > threshold).astype(int)
    
    return accuracy_score(binary_labels, binary_preds)

def calc_f1(preds, labels, threshold=0):
    """
    Calculate binary F1-score based on a threshold.
    
    Args:
        preds (np.ndarray): Predicted values.
        labels (np.ndarray): Ground truth values.
        threshold (float): Threshold to binarize predictions and labels.
    
    Returns:
        float: Binary F1-score.
    """
    binary_preds = (preds > threshold).astype(int)
    binary_labels = (labels > threshold).astype(int)
    
    return f1_score(binary_labels, binary_preds)

def calc_multiclass_metrics(preds, labels):
    """
    Calculate multiclass accuracy and weighted F1-score based on 7 sentiment classes (-3 to +3).
    
    Args:
        preds (np.ndarray): Predicted values.
        labels (np.ndarray): Ground truth values.
    
    Returns:
        dict: Dictionary containing 'multiclass_acc' and 'multiclass_f1'.
    """
    # Round to nearest integer and clip to [-3, 3] range
    rounded_preds = np.round(preds).clip(-3, 3)
    rounded_labels = np.round(labels).clip(-3, 3)
    
    # Convert to 7 classes (0-6 for -3 to +3)
    preds_classes = (rounded_preds + 3).astype(int)
    labels_classes = (rounded_labels + 3).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(labels_classes, preds_classes)
    f1 = f1_score(labels_classes, preds_classes, average="weighted")
    
    return {
        "multiclass_acc": acc,
        "multiclass_f1": f1
    }

def get_predictions(model, dataloader, device, output_csv_path=None):
    """
    Run inference on the provided dataloader and collect predictions, labels, and IDs.
    Optionally save results to a CSV file.

    Args:
        model (torch.nn.Module): Trained model for inference.
        dataloader (torch.utils.data.DataLoader): Dataloader to evaluate.
        device (str): Device to perform inference on ('cuda', 'cpu', etc.).
        output_csv_path (str, optional): Path to save the predictions as a CSV file.

    Returns:
        tuple: (np.ndarray of predictions, np.ndarray of labels, list of IDs)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            if isinstance(batch, dict):
                # Multimodal data
                inputs = {k: v.to(device) for k, v in batch.items() if k in ["text", "audio", "vision"]}
                labels = batch["label"].to(device)
                ids = batch["id"]
            else:
                # Unimodal data
                inputs, labels, ids, language = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                ids = ids

            # Forward pass
            outputs = model(inputs)
            
            # Collect predictions and labels
            #preds = outputs.squeeze().cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            
            # 添加 NaN 处理
            preds = np.where(np.isnan(preds), 0.0, preds)  # 将 NaN 替换为 0
            preds = np.where(np.isinf(preds), 0.0, preds)  # 将 Inf 替换为 0
            
            all_preds.append(preds)
            all_labels.append(labels)
            all_ids.append(ids)
    
    # Concatenate batch results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    if output_csv_path:
        with open(output_csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "predicted_class", "true_class"])
            for id_, pred, label in zip(all_ids, all_preds, all_labels):
                writer.writerow([id_, pred, label])
        print(f"Results saved to {output_csv_path}")

    return all_preds, all_labels

def evaluate_mosei(model, dataloader, device):
    """
    Evaluate model on CMU-MOSEI dataset using all relevant metrics.
    
    Args:
        model (torch.nn.Module): Trained model for evaluation.
        dataloader (torch.utils.data.DataLoader): Evaluation dataloader.
        device (str): Computation device.
    
    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    all_preds, all_labels = get_predictions(model, dataloader, device)
    
    # Calculate metrics

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    batch_correct = (all_preds == all_labels).sum()  # 预测正确的样本数
    batch_total = len(all_labels)  # 总样本数
    batch_accuracy = batch_correct / batch_total 
    '''
    下面是回归指标
    mae = calc_mae(all_preds, all_labels)
    corr = calc_correlation(all_preds, all_labels)
    acc = calc_binary_accuracy(all_preds, all_labels)
    f1 = calc_f1(all_preds, all_labels)
    
    # Multi-class metrics
    multiclass_metrics = calc_multiclass_metrics(all_preds, all_labels)
    
    # Combine all metrics
    metrics = {
        "mae": mae,
        "corr": corr,
        "binary_acc": acc,
        "binary_f1": f1,
        **multiclass_metrics
    }
    '''
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "batch_accuracy": batch_accuracy,
        "classification_report": class_report
    }
    return metrics

def log_metrics(metrics, split, epoch=None):
    """
    Log evaluation metrics to the logger.
    
    Args:
        metrics (dict): Dictionary of evaluation metrics.
        split (str): Dataset split ('train', 'val', 'test').
        epoch (int, optional): Epoch number (for training logs).
    """
    """
    epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
    logger.info(f"{epoch_str}{split.capitalize()} metrics:")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  Correlation: {metrics['corr']:.4f}")
    logger.info(f"  Binary Accuracy: {metrics['binary_acc']:.4f}")
    logger.info(f"  Binary F1: {metrics['binary_f1']:.4f}")
    logger.info(f"  7-class Accuracy: {metrics['multiclass_acc']:.4f}")
    logger.info(f"  7-class F1: {metrics['multiclass_f1']:.4f}")
    """

    epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
    logger.info(f"{epoch_str}{split.capitalize()} metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  Batch Accuracy: {metrics['batch_accuracy']:.4f}")  # 输出总体准确率

    # Log per-class metrics
    logger.info("  Classification Report:")
    for label, report in metrics["classification_report"].items():
        if isinstance(report, dict):  # Skip 'accuracy' key in classification_report
            logger.info(f"    Class {label}: Precision={report['precision']:.4f}, Recall={report['recall']:.4f}, F1={report['f1-score']:.4f}")