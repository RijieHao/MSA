import os
import csv
import subprocess
from pathlib import Path
from video2pkl.video2pkl.video2pkl import MOSEIExtractor

def generate_csv(video_dir, output_csv):
    """Generate a CSV file from the Test_Data directory"""
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
    # write header
        writer.writerow(["VideoDir", "VideoName", "Label", "Split", "ClassLabel"])
    # iterate over video directory
        for video_file in os.listdir(video_dir):
            if video_file.endswith(".mp4"):
                video_name = os.path.splitext(video_file)[0]  # remove file extension
                writer.writerow(["", video_name, -1, "test", "NEUT"])

def main():
    # path to video directory
    video_dir = "Test_Data"
    # generated CSV file path
    output_csv = "test_data.csv"
    # generated .pkl output directory
    output_pkl_dir = "MSAbypkl/data/data_pkl/test_pkl"

    # generate CSV file
    generate_csv(video_dir, output_csv)
    print(f"CSV file generated: {output_csv}")

    # initialize MOSEIExtractor
    processor = MOSEIExtractor(language="unknown")
    # call process_dataset to produce .pkl files
    processor.process_dataset(
        video_dir=video_dir,
        csv_path=output_csv,
        output_dir=output_pkl_dir,
        audio_dir=None  # do not use audio directory
    )
    print(f"Feature extraction complete, pkl files written to: {output_pkl_dir}")

    # run MSAbypkl\main.py and choose model evaluation
    main_script = "MSAbypkl/main.py"
    model_checkpoint = "best_models/en.pt"
    output_csv_path = "Test_Results/label_prediction.csv"

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # use subprocess to call main.py
    subprocess.run(
        ["python", main_script],
    )
    print(f"Model evaluation finished, results saved to: {output_csv_path}")

if __name__ == "__main__":
    main()