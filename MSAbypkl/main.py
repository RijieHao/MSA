import os
import sys
import logging
from pathlib import Path
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import csv

# Import global configurations
from config import (
    RAW_DATA_DIR, DATASET_NAME, DATASET_URL, PROCESSED_DATA_DIR, LOGS_DIR, MODELS_DIR,
    TEXT_MAX_LENGTH, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE,
    TEXT_EMBEDDING_DIM, SEED, DEVICE, EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_VAL,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, DROPOUT_RATE,
    HIDDEN_DIM, NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS, NUM_CLASSES
)

# Import necessary modules
from src.utils.logging import setup_logging
from src.data.dataset import get_dataloaders
from src.models.fusion import TransformerFusionModel
from src.training.trainer import Trainer
from src.utils.visualization import plot_training_curves, plot_scatter_predictions,plot_confusion_matrix
from src.training.metrics import log_metrics, get_predictions

# Add project root to path to enable module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def print_menu():
    """
    Display the main menu options for the console application.

    Shows the available actions the user can perform:
    - Model training
    - Model evaluation
    - Results visualization
    - Application exit
    """
    print("\n===== Multimodal Sentiment Analysis Console =====")
    print("1. Train a new model")
    print("2. Evaluate a trained model")
    print("3. Visualize training results")
    print("4. Exit")
    print("=================================================")

def train_model():
    """
    Handle model training workflow with configurable parameters.

    This function:
    1. Collects training parameters via user input
    2. Sets up logging and directories
    3. Loads and prepares the dataset
    4. Initializes the model and optimizer
    5. Handles checkpoint loading if specified
    6. Runs the training process
    7. Evaluates on test set
    8. Visualizes results

    All parameters can be customized through interactive prompts.
    """
    # Collect training parameters via user input
    checkpoint = input("Enter the path to the model checkpoint (.pt) or leave blank to train from scratch: ").strip()
    modalities = input("Enter modalities (default: text,audio,vision): ").strip() or "text,audio,vision"
    batch_size = int(input("Enter batch size (default: 32): ").strip() or BATCH_SIZE)
    num_epochs = int(input("Enter number of epochs (default: 15): ").strip() or NUM_EPOCHS)
    hidden_dim = int(input("Enter hidden dimension (default: 256): ").strip() or HIDDEN_DIM)
    num_layers = int(input("Enter number of layers (default: 4): ").strip() or NUM_TRANSFORMER_LAYERS)
    num_heads = int(input("Enter number of heads (default: 8): ").strip() or NUM_ATTENTION_HEADS)
    dropout = float(input("Enter dropout rate (default: 0.3): ").strip() or DROPOUT_RATE)
    lr = float(input("Enter learning rate (default: 1e-4): ").strip() or LEARNING_RATE)
    weight_decay = float(input("Enter weight decay (default: 1e-5): ").strip() or WEIGHT_DECAY)
    early_stopping = int(input("Enter early stopping patience (default: 10): ").strip() or EARLY_STOPPING_PATIENCE)
    gradient_clip = float(input("Enter gradient clipping value (default: 1.0): ").strip() or GRADIENT_CLIP_VAL)
    seed = int(input("Enter random seed (default: 42): ").strip() or SEED)
    log_dir = input("Enter log directory (default: logs/): ").strip() or LOGS_DIR
    save_dir = input("Enter model save directory (default: models/): ").strip() or MODELS_DIR
    device = input("Enter device (cuda/mps/cpu/auto): ").strip()

    # Setup logging
    logger = setup_logging(log_dir)

    if not device:
        device = DEVICE
    else:
        device = device.lower()
    if device not in ["cuda", "cpu", "auto"]:
        logger.error("Invalid device. Please enter  'cuda', or 'cpu'.")
        return
    if device == "auto":
        device = DEVICE

    # Get modalities to use
    modalities = modalities.split(",")

    # Log all training parameters
    logger.info("All hyperparameters set.")
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Hidden dimension: {hidden_dim}")
    logger.info(f"Number of transformer layers: {num_layers}")
    logger.info(f"Number of attention heads: {num_heads}")
    logger.info(f"Dropout rate: {dropout}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Early stopping patience: {early_stopping}")
    logger.info(f"Gradient clipping value: {gradient_clip}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Model save directory: {save_dir}")
    logger.info(f"Modalities: {modalities}")
    logger.info(f"Checkpoint: {checkpoint}")

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    logger.info(f"Random seed set to {seed}.")

    # Load data
    logger.info("\nLoading data...")
    dataloaders = get_dataloaders(
        modalities=modalities, 
        batch_size=batch_size,
        num_workers=0,
        #use_all_data=True,
        force_test_mode=True
    )

    # Create model
    logger.info("\nCreating model...")
    input_dims = {
        "language": TEXT_EMBEDDING_DIM,
        "acoustic": AUDIO_FEATURE_SIZE,
        "visual": VISUAL_FEATURE_SIZE
    }
    
    model = TransformerFusionModel(
        text_dim=input_dims["language"],
        audio_dim=input_dims["acoustic"],
        visual_dim=input_dims["visual"],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout,
        num_classes = NUM_CLASSES
    )
    model = model.to(device)

    # Log model summary
    logger.info(f"Model architecture:\n{model}")

    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )

    # Load checkpoint if provided
    start_epoch = 1
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            check = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(check["model_state_dict"])
            optimizer.load_state_dict(check["optimizer_state_dict"])
            start_epoch = check["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["valid"],
        test_loader=dataloaders["test"],
        device=device,
        log_dir=log_dir,
        model_dir=save_dir,
        experiment_name="multimodal_fusion",
    )

    logger.info("\nStarting training...")

    # Train the model
    logger.info("Starting training...")
    start_time = time.time()
    
    best_model, train_losses, val_losses, train_metrics, val_metrics = trainer.train(
        num_epochs=num_epochs,
        start_epoch=start_epoch
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Plot training curves
    plot_path = Path(log_dir) / "multimodal_training_curves.png"
    metrics = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics
    }
    plot_training_curves(
        train_losses, val_losses,
        save_path=plot_path
    )
    logger.info(f"Training curves plotted to {plot_path}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.test()
    log_metrics(test_metrics, "test")

    # Plot prediction scatter plot
    predictions, targets = get_predictions(model=best_model, dataloader=dataloaders["test"], device=device)
    scatter_path = Path(log_dir) / "multimodal_predictions.png"
    #if NUM_CLASSES==1:

        #plot_scatter_predictions(
        #    predictions, targets,
        #    save_path=scatter_path,
        #    title="Multimodal Sentiment Test Predictions vs Actual"
        #)
    #else:
    #    plot_confusion_matrix(targets, predictions, labels=["SNEG", "WNEG", "NEUT", "WPOS", "SPOS"], save_path=scatter_path)
    #)
    logger.info(f"Prediction scatter plot saved to {scatter_path}")
    
    logger.info("Training and evaluation completed!")


def evaluate_model():
    """
    Handle model evaluation workflow with pre-configured parameters.
    Automatically selects models based on language.
    """
    # 设置日志目录
    log_dir = LOGS_DIR
    logger = setup_logging(log_dir)
    logger.info("\nStarting evaluation...")

    # 设置模型权重路径
    zh_checkpoint = "best_models/zh.pt"  # 中文模型权重
    en_checkpoint = "best_models/en.pt"  # 英文模型权重

    # 设置设备
    device = DEVICE

    # 设置数据模态和批量大小
    modalities = ["text", "audio", "vision"]
    batch_size = BATCH_SIZE

    # 日志记录
    logger.info("Hyperparameters set.")
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Modalities: {modalities}")
    logger.info(f"Chinese checkpoint: {zh_checkpoint}")
    logger.info(f"English checkpoint: {en_checkpoint}")
    logger.info(f"Log directory: {log_dir}")

    # 加载数据
    dataloaders = get_dataloaders(
        modalities=modalities,
        batch_size=batch_size,
        num_workers=0,
        force_test_mode=True
    )
    test_loader = dataloaders["test"]
    logger.info("Data loaded.")

    # 定义输入维度
    input_dims = {
        "language": TEXT_EMBEDDING_DIM,
        "acoustic": AUDIO_FEATURE_SIZE,
        "visual": VISUAL_FEATURE_SIZE
    }

    # 初始化模型
    zh_model = TransformerFusionModel(
        text_dim=input_dims["language"],
        audio_dim=input_dims["acoustic"],
        visual_dim=input_dims["visual"],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_ATTENTION_HEADS,
        dropout_rate=DROPOUT_RATE,
        num_classes=NUM_CLASSES
    ).to(device)

    en_model = TransformerFusionModel(
        text_dim=input_dims["language"],
        audio_dim=input_dims["acoustic"],
        visual_dim=input_dims["visual"],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_ATTENTION_HEADS,
        dropout_rate=DROPOUT_RATE,
        num_classes=NUM_CLASSES
    ).to(device)

    # 加载模型权重
    logger.info(f"Loading Chinese model from checkpoint: {zh_checkpoint}")
    zh_checkpoint_data = torch.load(zh_checkpoint, map_location=device)
    zh_model.load_state_dict(zh_checkpoint_data["model_state_dict"])
    zh_model.eval()

    logger.info(f"Loading English model from checkpoint: {en_checkpoint}")
    en_checkpoint_data = torch.load(en_checkpoint, map_location=device)
    en_model.load_state_dict(en_checkpoint_data["model_state_dict"])
    en_model.eval()

    # 运行评估
    logger.info("Running evaluation on test set...")
    all_preds = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            # 获取批次数据
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items() if k in ["text", "audio", "vision"]}
                labels = batch["label"].to(device)
                ids = batch["id"]
                language = batch["language"]  # 获取语言属性
            else:
                inputs, labels, ids, language = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

            # 根据语言选择模型
            if language == "zh":
                outputs = zh_model(inputs)
            elif language == "en":
                outputs = en_model(inputs)
            else:
                logger.warning(f"Unknown language '{language}' for batch. Skipping...")
                continue

            # 收集预测和标签
            if NUM_CLASSES == 5:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                preds = outputs.squeeze().cpu().numpy()

            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_ids.extend(ids)

    # 保存预测结果到 CSV
    output_csv_path = "Test_Results/label_prediction.csv"
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Label"])
        for id_, pred in zip(all_ids, all_preds):
            writer.writerow([id_, pred])
    logger.info(f"Predictions saved to {output_csv_path}")

    # 计算并记录指标
    #from src.training.metrics import evaluate_mosei
    #test_metrics = evaluate_mosei(zh_model, test_loader, device)  # 使用中文模型计算指标
    #log_metrics(test_metrics, split="test")

    logger.info("Evaluation complete!")

def visualize_results():
    """
    Handle visualization of existing training results.
    
    This function:
    1. Locates saved visualization files in log directory
    2. Opens them using system default viewers
    
    Supports both training curves and prediction scatter plots.
    """
    log_dir = input("Enter log directory to visualize (default: logs/): ").strip() or LOGS_DIR
    plot_path = Path(log_dir) / "multimodal_training_curves.png"
    scatter_path = Path(log_dir) / "test_predictions.png"

    if plot_path.exists():
        logger.info(f"Opening training curves: {plot_path}")
        os.system(f"open '{plot_path}'" if sys.platform == "darwin" else f"xdg-open '{plot_path}'")
    else:
        logger.error("Training curves plot not found.")

    if scatter_path.exists():
        logger.info(f"Opening prediction scatter plot: {scatter_path}")
        os.system(f"open '{scatter_path}'" if sys.platform == "darwin" else f"xdg-open '{scatter_path}'")
    else:
        logger.error("Prediction scatter plot not found.")

def main():
    """
    Main entry point for the console application.
    
    Implements an interactive menu system that allows users to:
    - Train models
    - Evaluate models
    - Visualize results
    - Exit the application
    
    The menu runs in a loop until the user chooses to exit.
    """
    while True:
        print_menu()
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            train_model()
        elif choice == "2":
            evaluate_model()
        elif choice == "3":
            visualize_results()
        elif choice == "4":
            logger.info("Exiting the console. Goodbye!")
            break
        else:
            logger.error("Invalid choice. Please select 1-5.")
            continue

# Entry point
if __name__ == "__main__":
    main()