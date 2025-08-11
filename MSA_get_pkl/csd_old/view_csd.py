from mmsdk import mmdatasdk as md
import numpy as np
dataset1 = {}
dataset1["language"] = "our_MSA\ch_video_preprocess_debug\CMU_MOSEI_TimestampedWordVectors.csd"
dataset1["acoustic"] = "our_MSA\ch_video_preprocess_debug\CMU_MOSEI_COVAREP.csd"
dataset1["visual"] = "our_MSA\ch_video_preprocess_debug\CMU_MOSEI_VisualFacet42.csd"
dataset1["labels"] = "our_MSA\ch_video_preprocess_debug\CMU_MOSEI_Labels.csd"

dataset2 = {}
dataset2["language"] = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/data/raw/CMU_MOSEI/CMU_MOSEI_TimestampedWordVectors.csd"
dataset2["acoustic"] = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/data/raw/CMU_MOSEI/CMU_MOSEI_COVAREP.csd"
dataset2["visual"] = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/data/raw/CMU_MOSEI/CMU_MOSEI_VisualFacet42.csd"
dataset2["labels"] = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/data/raw/CMU_MOSEI/CMU_MOSEI_Labels.csd"


from mmsdk import mmdatasdk as md
import h5py  # 确保 h5py 已安装
def print_h5_structure(h5file, path="/", indent=0):
    """
    递归打印 HDF5 文件的层级结构，显示 Group 和 Dataset 的路径、keys 层级，以及 Dataset 的 shape 和 dtype
    :param h5file: HDF5 文件对象
    :param path: 当前路径
    :param indent: 缩进级别
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
    打印 CSD 文件的层级结构
    """
    with h5py.File(csd_file_path, "r") as h5file:
        print(f"打印 CSD 文件的层级结构: {csd_file_path}")
        print_h5_structure(h5file)

csd_file_path = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/data/raw/CMU_MOSEI/CMU_MOSEI_TimestampedWords.csd"
display_csd_structure(csd_file_path)
#mosei_dataset = md.mmdataset(dataset1)
mosei_dataset2 = md.mmdataset(dataset2)
print(111)