import os
import sys
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add the project root directory to the Python path so we can import modules from it
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import configuration variables
from config import (
    PROCESSED_DATA_DIR, DATASET_NAME, BATCH_SIZE,
    TEXT_EMBEDDING_DIM, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE
)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MOSEIDataset(Dataset):
    """
    A PyTorch Dataset class for the CMU-MOSEI dataset that supports
    loading data with multiple modalities (text, audio, vision) and IDs.

    Args:
        split (str): Dataset split to use ("train", "val", or "test").
        modalities (list): Modalities to include (default = all three).
    """
    def __init__(self, split="train", modalities=None):
        self.split = split
        self.modalities = modalities or ["text", "audio", "vision"]
        self.label_mapping = {
            "SNEG": 0,
            "WNEG": 1,
            "NEUT": 2,
            "WPOS": 3,
            "SPOS": 4
        }

        # Validate provided modalities
        for modality in self.modalities:
            if modality not in ["text", "audio", "vision"]:
                raise ValueError(f"Invalid modality: {modality}")

        # Construct path to dataset split
        self.data_path = PROCESSED_DATA_DIR / DATASET_NAME / f"{split}_data.pkl"
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. "
                f"Run preprocessing script first."
            )

        # Load the data for the split
        with open(self.data_path, "rb") as f:
            self.data = pickle.load(f)
        
        # Load feature dimension metadata (if available)
        metadata_path = PROCESSED_DATA_DIR / DATASET_NAME / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            # Fallback to default dimensions
            self.metadata = {
                "text_dim": TEXT_EMBEDDING_DIM,
                "audio_dim": AUDIO_FEATURE_SIZE,
                "visual_dim": VISUAL_FEATURE_SIZE
            }
        
        # Check if required fields are present
        optional_fields = ["labels", "id", "language","class_labels"]
        for field in optional_fields:
            if field not in self.data:
                logger.warning(f"Optional field '{field}' not found in data file: {self.data_path}")

        self.num_samples = len(self.data["labels"])
        logger.info(f"Loaded {self.num_samples} samples for {split} split")

    def __len__(self):
        """
        Return the total number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieve a sample (features + label + ID + language) by index.

        Args:
            idx (int): Sample index.
        Returns:
            dict: Dictionary with modality tensors, the label tensor, ID, and language.
        """
        sample = {}

        # Load each modality if available; otherwise, create a zero tensor
        for modality in self.modalities:
            if modality in self.data:
                sample[modality] = torch.tensor(self.data[modality][idx], dtype=torch.float32)
            else:
                logger.warning(f"Modality {modality} not found in data")
                # Create empty tensor with proper dimensions
                dim = self.metadata.get(f"{modality}_dim", 0)
                sample[modality] = torch.zeros(dim, dtype=torch.float32)
        

        if "class_labels" in self.data:
            label_idx = self.label_mapping.get(self.data["class_labels"][idx], -1)
            if label_idx == -1:
                logger.warning(f"Unknown class label '{self.data['class_labels'][idx]}' for sample {idx}")
                onehot = torch.zeros(len(self.label_mapping), dtype=torch.float32)
                sample["onehot"] = onehot
            else:
                onehot = torch.zeros(len(self.label_mapping), dtype=torch.float32)
                onehot[label_idx] = 1.0
                sample["onehot"] = onehot

        # Load label
        if "labels" in self.data:
            sample["label"] = torch.tensor(self.data["labels"][idx], dtype=torch.float32)

        # Load ID
        if "id" in self.data:
            sample["id"] = self.data["id"][idx]

        # Load language
        if "language" in self.data:
            sample["language"] = self.data["language"][idx]

        return sample

class MOSEIUnimodalDataset(Dataset):
    """
    A PyTorch Dataset class for loading a single modality (unimodal) from MOSEI.

    Args:
        split (str): Dataset split to use ("train", "val", or "test").
        modality (str): One of "text", "audio", or "vision".
    """
    def __init__(self, split="train", modality="text"):
        self.split = split
        self.modality = modality
        self.label_mapping = {
            "SNEG": 0,
            "WNEG": 1,
            "NEUT": 2,
            "WPOS": 3,
            "SPOS": 4
        }
        
        # Validate modality
        if modality not in ["text", "audio", "vision"]:
            raise ValueError(f"Invalid modality: {modality}")
        
        # Load dataset for the split
        self.data_path = PROCESSED_DATA_DIR / DATASET_NAME / f"{split}_data.pkl"
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. "
                f"Run preprocessing script first."
            )
        
        with open(self.data_path, "rb") as f:
            self.data = pickle.load(f)
        
        # Check that the modality, labels, id, and language exist
        required_fields = [modality, "labels", "id", "language"]
        optional_fields = ["class_labels"]

        for field in required_fields:
            if field not in self.data:
                raise ValueError(f"Field '{field}' not found in data file: {self.data_path}")

        for field in optional_fields:
            if field not in self.data:
                logger.warning(f"Optional field '{field}' not found in data file: {self.data_path}")

        self.num_samples = len(self.data["labels"])
        logger.info(f"Loaded {self.num_samples} samples for {split} split, modality: {modality}")
    
    def __len__(self):
        """
        Return the total number of samples.
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Retrieve a unimodal feature, label, ID, language, and onehot.

        Args:
            idx (int): Sample index.
        Returns:
            dict: Dictionary with the feature tensor, label tensor, ID, language, and onehot.
        """
        sample = {
            "feature": torch.tensor(self.data[self.modality][idx], dtype=torch.float32),
            "label": torch.tensor(self.data["labels"][idx], dtype=torch.float32),
            "id": self.data["id"][idx],
            "language": self.data["language"][idx]
        }

        # Load class_labels and create one-hot encoding
        if "class_labels" in self.data:
            class_label = self.data["class_labels"][idx]
            if isinstance(class_label, str):
                # Convert string label to index
                label_idx = self.label_mapping.get(class_label, -1)
                if label_idx == -1:
                    logger.warning(f"Unknown class label '{class_label}' for sample {idx}")
                    onehot = torch.zeros(len(self.label_mapping), dtype=torch.float32)
                else:
                    onehot = torch.zeros(len(self.label_mapping), dtype=torch.float32)
                    onehot[label_idx] = 1.0
            else:
                # Handle integer-based class_labels (if applicable)
                num_classes = max(self.data["class_labels"]) + 1  # Assuming class labels are 0-indexed
                onehot = torch.zeros(num_classes, dtype=torch.float32)
                onehot[class_label] = 1.0
        else:
            logger.warning(f"class_labels not found for sample {idx}")
            onehot = torch.zeros(len(self.label_mapping), dtype=torch.float32)  # Empty tensor if class_labels is missing

        sample["onehot"] = onehot
        
        return sample

def get_dataloaders(modalities=None, batch_size=BATCH_SIZE, num_workers=2):
    """
    Create multimodal dataloaders for train, val, and test splits.

    Args:
        modalities (list): Modalities to include. Default is all three.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker threads for loading data.

    Returns:
        dict: Dataloaders for each split.
    """
    dataloaders = {}
    
    # Iterate over the splits
    for split in ["train", "val", "test"]:
        dataset = MOSEIDataset(split=split, modalities=modalities)
        
        shuffle = (split == "train") # Only shuffle for training
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders

def get_unimodal_dataloaders(modality, batch_size=BATCH_SIZE, num_workers=2):
    """
    Create unimodal dataloaders for train, val, and test splits.

    Args:
        modality (str): One of "language", "acoustic", or "visual".
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker threads for loading data.

    Returns:
        dict: Dataloaders for each split.
    """
    dataloaders = {}
    
    # Iterate over the splits
    for split in ["train", "val", "test"]:
        dataset = MOSEIUnimodalDataset(split=split, modality=modality)
        
        shuffle = (split == "train") # Only shuffle for training
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders
