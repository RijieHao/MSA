
import numpy as np
import h5py  # ensure h5py is installed
from mmsdk import mmdatasdk as md

dataset1 = {}
dataset1["language"] = r"our_MSA\ch_video_preprocess_debug\CMU_MOSEI_TimestampedWordVectors.csd"
dataset1["acoustic"] = r"our_MSA\ch_video_preprocess_debug\CMU_MOSEI_COVAREP.csd"
dataset1["visual"] = r"our_MSA\ch_video_preprocess_debug\CMU_MOSEI_VisualFacet42.csd"
dataset1["labels"] = r"our_MSA\ch_video_preprocess_debug\CMU_MOSEI_Labels.csd"

dataset2 = {}
dataset2["language"] = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/data/raw/CMU_MOSEI/CMU_MOSEI_TimestampedWordVectors.csd"
dataset2["acoustic"] = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/data/raw/CMU_MOSEI/CMU_MOSEI_COVAREP.csd"
dataset2["visual"] = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/data/raw/CMU_MOSEI/CMU_MOSEI_VisualFacet42.csd"
dataset2["labels"] = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/data/raw/CMU_MOSEI/CMU_MOSEI_Labels.csd"
def print_h5_structure(h5file, path="/", indent=0):
    """
    Recursively print the HDF5 file hierarchical structure showing Group and Dataset paths,
    the keys hierarchy, and each Dataset's shape and dtype.
    :param h5file: HDF5 file object
    :param path: current path
    :param indent: indentation level
    """
    obj = h5file[path]
    if isinstance(obj, h5py.Group):
        print("  " * indent + f"Group: {path}")
        for key in obj.keys():
            print_h5_structure(h5file, f"{path}/{key}", indent + 1)
    elif isinstance(obj, h5py.Dataset):
        print("  " * indent + f"Dataset: {path}, shape: {obj.shape}, dtype: {obj.dtype}")

def display_csd_structure(csd_file_path):
    """
    Print the CSD file's hierarchical structure
    """
    with h5py.File(csd_file_path, "r") as h5file:
    print(f"Print CSD file hierarchical structure: {csd_file_path}")
    print_h5_structure(h5file)

csd_file_path = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/data/raw/CMU_MOSEI/CMU_MOSEI_TimestampedWords.csd"
display_csd_structure(csd_file_path)
#mosei_dataset = md.mmdataset(dataset1)
mosei_dataset2 = md.mmdataset(dataset2)
print(111)