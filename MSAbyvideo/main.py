import os
import sys
import logging
from pathlib import Path
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

# Import global configurations
from config import (
    LOGS_DIR, MODELS_DIR, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE,
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
    print("\n===== Multimodal Sentiment Analysis Console =====")
    print("1. Train a new model")
    print("2. Evaluate a trained model")
    print("3. Visualize training results")
    print("4. Exit")
    print("=================================================")

def train_model():
    # Collect training parameters via user input
    video_dir = input("Enter the path to the video directory: ").strip()
    audio_dir = input("Enter the path to the audio directory (optional): ").strip() or None
    csv_path = input("Enter the path to the CSV file: ").strip()
    checkpoint = input("Enter the path to the model checkpoint (.pt) or leave blank to train from scratch: ").strip()
    batch_size = int(input("Enter batch size (default: 32): ").strip() or BATCH_SIZE)
    all_test = input("Use all data as test set? (yes/no, default: no): ").strip().lower() == "yes"
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
    device = input("Enter device (cuda/cpu/auto): ").strip()

    # Setup logging
    logger = setup_logging(log_dir)

    if not device:
        device = DEVICE
    else:
        device = device.lower()
    if device not in ["cuda", "cpu", "auto"]:
        logger.error("Invalid device. Please enter 'cuda', or 'cpu'.")
        return
    if device == "auto":
        device = DEVICE


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
        video_dir=video_dir,
        csv_path=csv_path,
        batch_size=batch_size,
        audio_dir=audio_dir,
        all_test=all_test
    )

    # Create model
    logger.info("\nCreating model...")
    
    model = TransformerFusionModel(
        text_dim=TEXT_EMBEDDING_DIM,
        audio_dim=AUDIO_FEATURE_SIZE,
        visual_dim=VISUAL_FEATURE_SIZE,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout,
        num_classes=NUM_CLASSES
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
            check = torch.load(checkpoint_path)
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
    plot_training_curves(
        train_losses, val_losses,
        save_path=plot_path
    )
    logger.info(f"Training curves plotted to {plot_path}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.test()
    log_metrics(test_metrics, "test")
    
    logger.info("Training and evaluation completed!")


def evaluate_model():

    video_dir = input("Enter the path to the video directory: ").strip()
    audio_dir = input("Enter the path to the audio directory (optional): ").strip() or None
    csv_path = input("Enter the path to the CSV file: ").strip()
    all_test = input("Use all data as test set? (yes/no, default: no): ").strip().lower() == "yes"


    # Setup logging with user-specified directory
    log_dir = input("Enter log directory (default: logs/): ").strip() or LOGS_DIR
    logger = setup_logging(log_dir)
    logger.info("\nStarting evaluation...")

    # Get required checkpoint path
    checkpoint = input("Enter the path to the model checkpoint (.pt): ").strip()
    if not checkpoint:
        logger.error("Checkpoint path is required for evaluation.")
        return
    # Collect evaluation parameters
    batch_size = input("Enter batch size (default: 32): ").strip() or BATCH_SIZE
    
    # Validate and set device
    device = input("Enter device (cuda/mps/cpu/default: auto): ").strip()
    if not device:
        device = DEVICE
    else:
        device = device.lower()
    if device not in ["cuda", "cpu", "auto"]:
        logger.error("Invalid device. Please enter 'cuda', or 'cpu'.")
        return
    if device == "auto":
        device = DEVICE


    # Load data
    dataloaders = get_dataloaders(
        video_dir=video_dir,
        csv_path=csv_path,
        batch_size=batch_size,
        audio_dir=audio_dir,
        all_test=all_test
    )
    test_loader = dataloaders["test"]
    logger.info("Data loaded.")

    # Initialize model
    model = TransformerFusionModel(
        text_dim=TEXT_EMBEDDING_DIM,
        audio_dim=AUDIO_FEATURE_SIZE,
        visual_dim=VISUAL_FEATURE_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_ATTENTION_HEADS,
        dropout_rate=DROPOUT_RATE,
        num_classes=NUM_CLASSES
    )
    model = model.to(device)
    logger.info("Model initialized.")

    # Load checkpoint
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    check = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(check["model_state_dict"])
    model.eval()

    # Run evaluation
    logger.info("Running evaluation on test set...")
    predictions, targets = get_predictions(model=model, dataloader=test_loader, device=device, output_csv_path="MSAbypkl/logs/log.csv")

    # Compute and log metrics
    from src.training.metrics import evaluate_mosei
    test_metrics = evaluate_mosei(model, test_loader, device)
    log_metrics(test_metrics, split="test")
    
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