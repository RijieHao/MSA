import os
import pickle
from pathlib import Path
import numpy as np

def inspect_pkl_structure(pkl_dir):
    """
    检查指定文件夹中的 .pkl 文件结构并打印内容摘要。
    
    Args:
        pkl_dir (str): 包含 .pkl 文件的文件夹路径。
    """
    pkl_dir = Path(pkl_dir)
    if not pkl_dir.exists():
        print(f"❌ 文件夹 {pkl_dir} 不存在！")
        return

    # 遍历文件夹中的所有 .pkl 文件
    for pkl_file in pkl_dir.glob("*.pkl"):
        print(f"\n📂 正在检查文件: {pkl_file.name}")
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            # 打印文件内容摘要
            if isinstance(data, dict):
                print(f"✅ 文件内容为字典，包含以下键：{list(data.keys())}")
                for key, value in data.items():
                    if isinstance(value, list):
                        print(f"  - {key}: 列表，长度为 {len(value)}")
                        if len(value) > 0:
                            print(f"    示例数据类型: {type(value[0])}")
                            if isinstance(value[0], (list, np.ndarray)):
                                print(f"    示例数据形状: {np.array(value[0]).shape}")
                    else:
                        print(f"  - {key}: 类型为 {type(value)}")
            else:
                print(f"⚠️ 文件内容不是字典，类型为 {type(data)}")

        except Exception as e:
            print(f"❌ 无法读取文件 {pkl_file.name}，错误信息：{e}")

if __name__ == "__main__":
    # 设置包含 .pkl 文件的文件夹路径
    pkl_directory = "Multimodal-Sentiment-Analysis-with-MOSEI-Dataset\data\processed\CMU_MOSEI"  # 替换为您的 .pkl 文件夹路径
    inspect_pkl_structure(pkl_directory)