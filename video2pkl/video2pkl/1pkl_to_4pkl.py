import pickle
import os


TEXT_EMBEDDING_DIM = 768
AUDIO_FEATURE_SIZE = 40
VISUAL_FEATURE_SIZE = 35
SEED = 42
# Configure paths
input_pkl_path = "E:/kaggle/MSAbypkl/unaligned_en.pkl"  # replace with your input pkl file path
output_dir = "E:/kaggle/MSAbypkl/data/data_pkl/unaligned_en_pkl"   # replace with your output folder path
os.makedirs(output_dir, exist_ok=True)

# Load original .pkl file
with open(input_pkl_path, "rb") as f:
    data = pickle.load(f)

# Check that data contains train, valid, test
required_keys = ["train", "valid", "test"]
for key in required_keys:
    if key not in data:
        raise ValueError(f"Missing key '{key}' in the input .pkl file.")

# Save train, val, test into separate .pkl files
for split in required_keys:
    split_path = os.path.join(output_dir, f"{split}_data.pkl")
    with open(split_path, "wb") as f:
        pickle.dump(data[split], f)
    print(f"Saved {split} data to {split_path}")

# Generate metadata.pkl file
metadata = {
    "text_dim": TEXT_EMBEDDING_DIM,
    "audio_dim": AUDIO_FEATURE_SIZE,
    "visual_dim": VISUAL_FEATURE_SIZE,
    "num_classes": 1,
    "train_samples": len(data["train"]["regression_labels"]),
    "val_samples": len(data["valid"]["regression_labels"]),
    "test_samples": len(data["test"]["regression_labels"]),
}

metadata_path = os.path.join(output_dir, "metadata.pkl")
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
print(f"Saved metadata to {metadata_path}")