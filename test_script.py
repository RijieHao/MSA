import os
import csv
import subprocess
from pathlib import Path
from video2pkl.video2pkl.video2pkl import MOSEIExtractor

def generate_csv(video_dir, output_csv):
    """生成 CSV 文件"""
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["VideoDir", "VideoName", "Label", "Split", "ClassLabel"])
        # 遍历视频文件夹
        for video_file in os.listdir(video_dir):
            if video_file.endswith(".mp4"):
                video_name = os.path.splitext(video_file)[0]  # 去掉后缀
                writer.writerow([video_dir, video_name, -1, "test", "WENG"])

def update_config(dataset_name):
    """更新 MSAbypkl/config.py 中的 DATASET_NAME"""
    config_path = Path("e:/kaggle/MSAbypkl/config.py")
    with open(config_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    with open(config_path, "w", encoding="utf-8") as file:
        for line in lines:
            if line.startswith("DATASET_NAME"):
                file.write(f'DATASET_NAME = "{dataset_name}"\n')
            else:
                file.write(line)

def main():
    # 视频文件夹路径
    video_dir = "path_to_your_video_folder"  # 替换为你的视频文件夹路径
    # 生成的 CSV 文件路径
    output_csv = "generated_test.csv"
    # 生成的 .pkl 文件输出目录
    output_pkl_dir = "e:/kaggle/MSAbypkl/data/data_pkl/test_generated_pkl"

    # 生成 CSV 文件
    generate_csv(video_dir, output_csv)
    print(f"CSV 文件已生成: {output_csv}")

    # 初始化 MOSEIExtractor
    processor = MOSEIExtractor(language="unknown")
    # 调用 process_dataset 方法生成 .pkl 文件
    processor.process_dataset(
        video_dir=video_dir,
        csv_path=output_csv,
        output_dir=output_pkl_dir,
        audio_dir=None  # 不使用音频文件夹
    )
    print(f"特征提取完成，pkl 文件已生成到: {output_pkl_dir}")

    # 更新 config.py 中的 DATASET_NAME
    dataset_name = f"data/data_pkl/test_generated_pkl"
    update_config(dataset_name)
    print(f"已更新 MSAbypkl/config.py 中的 DATASET_NAME 为: {dataset_name}")

    # 运行 MSAbypkl/main.py
    print("开始运行 MSAbypkl/main.py...")
    subprocess.run(["python", "e:/kaggle/MSAbypkl/main.py"], check=True)

if __name__ == "__main__":
    main()