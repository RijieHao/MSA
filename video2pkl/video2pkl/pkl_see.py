import os
import pickle
from pathlib import Path
import numpy as np


def inspect_pkl_structure(pkl_dir):
    """Inspect .pkl files in a directory and print a brief summary.

    Args:
        pkl_dir (str): Path to the folder containing .pkl files.
    """

    pkl_dir = Path(pkl_dir)
    if not pkl_dir.exists():
        print(f"❌ Folder {pkl_dir} does not exist!")
        return

    # Iterate over all .pkl files in the folder
    for pkl_file in pkl_dir.glob("*.pkl"):
        print(f"\n📂 Inspecting file: {pkl_file.name}")
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            # Print a summary of the file contents
            if isinstance(data, dict):
                print(f"✅ File is a dict with keys: {list(data.keys())}")
                for key, value in data.items():
                    if isinstance(value, list):
                        print(f"  - {key}: list, length {len(value)}")
                        if len(value) > 0:
                            print(f"    Example data type: {type(value[0])}")
                            if isinstance(value[0], (list, np.ndarray)):
                                print(f"    Example data shape: {np.array(value[0]).shape}")
                    else:
                        print(f"  - {key}: type {type(value)}")
            else:
                print(f"⚠️ File content is not a dict, type {type(data)}")

        except Exception as e:
            print(f"❌ Failed to read file {pkl_file.name}, error: {e}")


if __name__ == "__main__":
    # Set path to the folder that contains .pkl files
    pkl_directory = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset\\data\\processed\\CMU_MOSEI"  # Replace with your .pkl directory path
    inspect_pkl_structure(pkl_directory)