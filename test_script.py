import os
import csv
import subprocess
from pathlib import Path
from video2pkl.video2pkl.video2pkl import MOSEIExtractor

def generate_csv(video_dir, output_csv):
    """从 Test_Data 文件夹生成 CSV 文件"""
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["VideoDir", "VideoName", "Label", "Split", "ClassLabel"])
        # 遍历视频文件夹
        for video_file in os.listdir(video_dir):
            if video_file.endswith(".mp4"):
                video_name = os.path.splitext(video_file)[0]  # 去掉后缀
                writer.writerow([video_dir, video_name, -1, "test", "NEUT"])

def main():
    # 视频文件夹路径
    video_dir = "Test_Data"  # Test_Data 文件夹路径
    # 生成的 CSV 文件路径
    output_csv = "test_data.csv"
    # 生成的 .pkl 文件输出目录
    output_pkl_dir = "e:/kaggle/MSAbypkl/data/data_pkl/test_pkl"

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

    # 运行 MSAbypkl\main.py 并选择模型评估
    main_script = "e:/kaggle/MSAbypkl/main.py"
    model_checkpoint = "best_models/en.pt"
    output_csv_path = "Test_Results/label_prediction.csv"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 使用 subprocess 调用 main.py 并传递参数
    subprocess.run(
        [
            "python", main_script,
            "--choice", "2",  # 选择评估模式
            "--checkpoint", model_checkpoint,  # 模型权重路径
            "--output_csv", output_csv_path  # 固定 CSV 输出路径
        ],
        check=True
    )
    print(f"模型评估完成，结果已保存到: {output_csv_path}")

if __name__ == "__main__":
    main()