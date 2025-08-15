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
    TEXT_EMBEDDING_DIM, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE, NUM_CLASSES
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
    def __init__(self, split="train", modalities=None, use_all_data=False, force_test_mode=False):
        self.split = split
        self.modalities = modalities or ["text", "audio", "vision"]
        self.label_mapping = {
            "SNEG": 0,
            "WNEG": 1,
            "NEUT": 2,
            "WPOS": 3,
            "SPOS": 4
        }
        self.label_mapping2 = {
            -1: 0,
            -0.8: 0,
            -0.6: 1,
            -0.4: 1,
            -0.2: 1,
            0: 2,
            0.2: 3,
            0.4: 3,
            0.6: 3,
            0.8: 4,
            1: 4,
        }
        

        # Validate provided modalities

        for modality in self.modalities:
            if modality not in ["text", "audio", "vision"]:
                raise ValueError(f"Invalid modality: {modality}")

        # Construct paths to dataset splits
        base_path = PROCESSED_DATA_DIR / DATASET_NAME
        train_path = base_path / "train_data.pkl"
        valid_path = base_path / "valid_data.pkl"
        test_path = base_path / "test_data.pkl"



        if use_all_data and split == "train":
            # 合并 train, valid, test 数据
            self.data = self._load_and_merge_data([train_path, valid_path, test_path])
        elif force_test_mode and split in ["test"]:
            # 强制按 test 模式读取
            self.data = self._load_and_merge_data([test_path])
        else:
            # 正常加载指定 split 的数据

            self.data_path = base_path / f"{split}_data.pkl"
            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"Data file not found: {self.data_path}. "
                    f"Run preprocessing script first."
                )
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)

        # Load feature dimension metadata (if available)
        metadata_path = base_path / "metadata.pkl"
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
        #optional_fields = ["labels", "id", "language","class_labels"]
        optional_fields = ["id", "class_labels","language"]
        for field in optional_fields:
            if field not in self.data:
                logger.warning(f"Optional field '{field}' not found in data")
        if "class_labels" in self.data:
            self.num_samples = len(self.data["class_labels"])
        elif "regression_labels" in self.data:
            self.num_samples = len(self.data["regression_labels"])
        logger.info(f"Loaded {self.num_samples} samples for {split} split")

    def _load_and_merge_data(self, paths):
        """Load and merge data from multiple pkl files."""
        merged_data = {}
        for path in paths:
            if path.exists():
                with open(path, "rb") as f:
                    data = pickle.load(f)
                for key, value in data.items():
                    if key not in merged_data:
                        merged_data[key] = value
                    else:
                        if isinstance(merged_data[key], list):
                            merged_data[key].extend(value)
                        elif isinstance(merged_data[key], np.ndarray):
                            merged_data[key] = np.concatenate((merged_data[key], value), axis=0)
            else:
                logger.warning(f"Data file not found: {path}")
        return merged_data
    def map_to_label3(self, value):
        """
        Map a continuous value to a label based on the following ranges:
        -3 ~ -2 -> 0
        -2 ~ -0.5 -> 1
        -0.5 ~ 0.5 -> 2
        0.5 ~ 2 -> 3
        2 ~ 3 -> 4

        Args:
            value (float): The continuous value to map.

        Returns:
            int: The mapped label.
        """
        if -3 <= value < -2:
            return 0
        elif -2 <= value < -0.5:
            return 1
        elif -0.5 <= value < 0.5:
            return 2
        elif 0.5 <= value < 2:
            return 3
        elif 2 <= value <= 3:
            return 4
        else:
            logger.warning(f"Value {value} is out of range for mapping")
            return -1  # Return -1 for out-of-range values

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
        modality_dims = {
            "text": TEXT_EMBEDDING_DIM,
            "audio": AUDIO_FEATURE_SIZE,
            "vision": VISUAL_FEATURE_SIZE
        }

        # Load each modality if available; otherwise, create a zero tensor
        for modality in self.modalities:
            target_dim = modality_dims.get(modality, 0)
            if modality in self.data:
                modality_data = self.data[modality][idx]
                # 检查是否包含 NaN 值，如果有则替换为零
                if np.isnan(modality_data).any():
                    logger.warning(f"Modality {modality} for sample {idx} contains NaN values. Replacing with zeros.")
                    modality_data = np.nan_to_num(modality_data, nan=0.0)
                modality_data = np.array(modality_data)
                # 截断或补零到目标维度
                if modality_data.shape[-1] > target_dim:
                    modality_data = modality_data[..., :target_dim]
                elif modality_data.shape[-1] < target_dim:
                    pad_width = target_dim - modality_data.shape[-1]
                    modality_data = np.pad(modality_data, (0, pad_width), mode='constant')
                sample[modality] = torch.tensor(modality_data, dtype=torch.float32)
            else:
                logger.warning(f"Modality {modality} not found in data")
                sample[modality] = torch.zeros(target_dim, dtype=torch.float32)
        

        if "class_labels" in self.data:
            class_label = self.data["class_labels"][idx]
            label_idx = self.label_mapping.get(class_label, -1)
            if label_idx == -1:
                logger.warning(f"Unknown class label '{class_label}' for sample {idx}")
                sample["label"] = label_idx
            else:
                sample["label"] = label_idx
        elif "regression_labels" in self.data:
            if NUM_CLASSES == 1:
                sample["label"] = torch.tensor(self.data["regression_labels"][idx], dtype=torch.float32)
            elif NUM_CLASSES == 5:
                regression_label = self.data["regression_labels"][idx]
                mapped_label = self.label_mapping2.get(regression_label, -1)
                #mapped_label = self.map_to_label3(regression_label)   # 映射到 label_mapping2
                if mapped_label == -1:
                    logger.warning(f"Unknown regression label '{regression_label}' for sample {idx}")
                sample["label"] = torch.tensor(mapped_label, dtype=torch.long)  # 转为分类任务的整数标签


        # Load label
        #if "labels" in self.data:
        #    sample["label"] = torch.tensor(self.data["labels"][idx], dtype=torch.float32)

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
        #optional_fields = ["labels", "id", "language","class_labels"]
        optional_fields = ["id", "class_labels", "language"]

        for field in optional_fields:
            if field not in self.data:
                logger.warning(f"Optional field '{field}' not found in data file: {self.data_path}")

        self.num_samples = len(self.data["class_labels"])
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
        sample = {}

        sample["feature"] = torch.tensor(self.data[self.modality][idx], dtype=torch.float32)

        if "class_labels" in self.data:
            class_label = self.data["class_labels"][idx]
            label_idx = self.label_mapping.get(class_label, -1)
            if label_idx == -1:
                logger.warning(f"Unknown class label '{class_label}' for sample {idx}")
                sample["label"] = label_idx
            else:
                sample["label"] = label_idx

        # Load label
        #if "labels" in self.data:
        #    sample["label"] = torch.tensor(self.data["labels"][idx], dtype=torch.float32)

        # Load ID
        if "id" in self.data:
            sample["id"] = self.data["id"][idx]

        # Load language
        if "language" in self.data:
            sample["language"] = self.data["language"][idx]
        
        return sample

def get_dataloaders(modalities=None, batch_size=BATCH_SIZE, num_workers=2, use_all_data=False, force_test_mode=False):
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
    if force_test_mode == False:
        train_dataset = MOSEIDataset(
        split="train",
        modalities=modalities,
        use_all_data=use_all_data,  # 合并 train, valid, test 数据
        force_test_mode=False  # 不影响 valid 和 test
        )
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # 训练数据需要打乱
            num_workers=num_workers,
            pin_memory=True
        )
        valid_dataset = MOSEIDataset(
            split="valid",
            modalities=modalities,
            use_all_data=False,  # 确保 valid 数据不变
            force_test_mode=False  # 不影响 test
        )
        dataloaders["valid"] = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证数据不需要打乱
            num_workers=num_workers,
            pin_memory=True
        )
    # Test split
    test_dataset = MOSEIDataset(
        split="test",
        modalities=modalities,
        use_all_data=False,  # 确保 test 数据不变
        force_test_mode=force_test_mode  # 如果需要强制按 test 模式读取
    )
    dataloaders["test"] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试数据不需要打乱
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
