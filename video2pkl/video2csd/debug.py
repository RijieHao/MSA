import os
import sys
import pandas as pd
from pathlib import Path
import traceback
import h5py

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入我们的模块
from get_3m import process_video_dataset_to_accurate_mosei_from_csv, AccurateMOSEIExtractor

def debug_csv_structure(csv_path):
    """调试CSV文件结构"""
    print("=" * 60)
    print("🔍 CSV文件结构调试")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        print("📁 当前目录下的文件:")
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        for file in csv_files:
            print(f"  - {file}")
        return False
    
    try:
        print(f"📊 读取CSV文件: {csv_path}")
        df = pd.read_csv(csv_path, dtype={0: str, 1: str})
        print(f"✅ 文件读取成功: {csv_path}")
        print(f"📊 数据形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")
        print(f"📋 列名: {list(df.columns)}")
        print("\n📑 前3行数据:")
        print(df.head(3))
        print("\n🔢 数据类型:")
        print(df.dtypes)
        
        print("\n🔍 路径格式检查:")
        for i in range(min(3, len(df))):
            video_id = str(df.iloc[i, 0])
            clip_id = str(df.iloc[i, 1])
            print(f"  行{i}: video_id='{video_id}', clip_id='{clip_id}' -> 路径: {video_id}/{clip_id}.mp4")
        
        if df.shape[1] >= 6:
            label_columns = ['overall', 'text', 'audio', 'visual']
            for i, col_name in enumerate(label_columns):
                col_idx = i + 2
                if col_idx < df.shape[1]:
                    values = df.iloc[:, col_idx]
                    values = pd.to_numeric(values, errors='coerce')
                    values = values.dropna()
                    if len(values) > 0:
                        print(f"📈 {col_name}_label (列{col_idx}): 范围=[{values.min():.3f}, {values.max():.3f}], 均值={values.mean():.3f}")
                    else:
                        print(f"📈 {col_name}_label (列{col_idx}): 无有效数值数据")
        return True
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        traceback.print_exc()
        return False

def debug_video_files(file_path, video_base_dir):
    """调试视频文件存在性"""
    print("\n" + "=" * 60)
    print("🎬 视频文件存在性调试")
    print("=" * 60)
    
    video_base_dir = Path(video_base_dir)
    print(f"📁 视频基础目录: {video_base_dir}")
    print(f"📁 目录是否存在: {video_base_dir.exists()}")
    
    if not video_base_dir.exists():
        print("❌ 视频基础目录不存在!")
        return False
    
    subdirs = [d for d in video_base_dir.iterdir() if d.is_dir()]
    print(f"📂 找到 {len(subdirs)} 个子目录:")
    for subdir in subdirs[:5]:
        print(f"  - {subdir.name}")
    
    try:
        df = pd.read_csv(file_path, dtype={0: str, 1: str})
        found_videos = 0
        missing_videos = 0
        print(f"\n🔍 检查前10个视频文件...")
        for idx, row in df.head(10).iterrows():
            video_id = str(row.iloc[0]).strip()
            clip_id = str(row.iloc[1]).strip()
            video_path = video_base_dir / video_id / f"{clip_id}.mp4"
            print(f"  🔍 检查: video_id='{video_id}', clip_id='{clip_id}'")
            print(f"     构建路径: {video_path}")
            if video_path.exists():
                file_size = video_path.stat().st_size / (1024 * 1024)
                print(f"     ✅ 找到文件 - {file_size:.2f}MB")
                found_videos += 1
            else:
                print(f"     ❌ 文件不存在")
                parent_dir = video_path.parent
                if parent_dir.exists():
                    files_in_dir = list(parent_dir.glob("*.mp4"))
                    print(f"     📂 父目录存在，包含{len(files_in_dir)}个MP4文件:")
                    for f in files_in_dir[:3]:
                        print(f"         - {f.name}")
                else:
                    print(f"     📂 父目录不存在: {parent_dir}")
                missing_videos += 1
        print(f"\n📊 视频文件统计 (前10个):")
        print(f"  ✅ 找到: {found_videos}")
        print(f"  ❌ 缺失: {missing_videos}")
        return found_videos > 0
    except Exception as e:
        print(f"❌ 检查视频文件时出错: {e}")
        return False

def debug_feature_extraction(file_path, video_base_dir):
    """调试特征提取过程"""
    print("\n" + "=" * 60)
    print("🔧 特征提取过程调试")
    print("=" * 60)
    try:
        print("🚀 初始化AccurateMOSEIExtractor...")
        extractor = AccurateMOSEIExtractor(language="zh")
        print("✅ 提取器初始化成功")
        df = pd.read_csv(file_path, dtype={0: str, 1: str})
        video_base_dir = Path(video_base_dir)
        test_video_path = None
        for idx, row in df.iterrows():
            video_id = str(row.iloc[0]).strip()
            clip_id = str(row.iloc[1]).strip()
            video_path = video_base_dir / video_id / f"{clip_id}.mp4"
            print(f"🔍 检查视频: {video_id}/{clip_id} -> {video_path}")
            if video_path.exists():
                test_video_path = video_path
                print(f"🎯 选择测试视频: {video_path}")
                break
        if test_video_path is None:
            print("❌ 没有找到可用的测试视频")
            return False
        print(f"\n🔍 测试特征提取...")
        try:
            print("📝 1. 测试词特征提取...")
            word_features = extractor.extract_word_features(str(test_video_path))
            print(f"   ✅ 词特征: {len(word_features)} 个词")
            if word_features:
                print(f"   📊 第一个词特征形状: {word_features[0][0].shape if len(word_features[0]) > 0 else 'None'}")
        except Exception as e:
            print(f"   ❌ 词特征提取失败: {e}")
            traceback.print_exc()
        try:
            print("🔊 2. 测试音频特征提取...")
            audio_features = extractor.extract_covarep_acoustic_features(str(test_video_path))
            print(f"   ✅ 音频特征形状: {audio_features.shape}")
            print(f"   📊 特征范围: [{audio_features.min():.3f}, {audio_features.max():.3f}]")
        except Exception as e:
            print(f"   ❌ 音频特征提取失败: {e}")
            traceback.print_exc()
        try:
            print("👁️ 3. 测试视觉特征提取...")
            visual_features = extractor.extract_openface_visual_features(str(test_video_path))
            print(f"   ✅ 视觉特征形状: {visual_features.shape}")
            print(f"   📊 特征范围: [{visual_features.min():.3f}, {visual_features.max():.3f}]")
        except Exception as e:
            print(f"   ❌ 视觉特征提取失败: {e}")
            traceback.print_exc()
        return True
    except Exception as e:
        print(f"❌ 特征提取调试失败: {e}")
        traceback.print_exc()
        return False

def debug_dataset_creation():
    """调试数据集创建过程"""
    print("\n" + "=" * 60)
    print("📦 数据集创建过程调试")
    print("=" * 60)
    file_path = "our_MSA/meta_test_only_debug.csv"
    video_base_dir = "our_MSA/ch_video_debug"
    output_dir = "our_MSA/ch_video_preprocess_debug"
    try:
        print("🚀 开始处理数据集...")
        dataset = process_video_dataset_to_accurate_mosei_from_csv(
            csv_path=file_path,
            video_base_dir=video_base_dir,
            output_dir=output_dir,
            language="zh"
        )
        return dataset, output_dir
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        traceback.print_exc()
        return None, None

def main():
    """主调试函数"""
    print(" 全面调试 get_3m.py")
    print("=" * 80)
    
    # 查找数据文件
    data_file = "our_MSA/meta_test_only_debug.csv"
    
    video_base_dir = "our_MSA/ch_video_debug"
    '''
    # 1. 调试文件结构
    if not debug_csv_structure(data_file):
        print("⛔ 数据文件调试失败，停止后续调试")
        return
    
    # 2. 调试视频文件
    if not debug_video_files(data_file, video_base_dir):
        print("⚠️ 视频文件存在问题，但继续调试...")
    
    # 3. 调试特征提取
    if not debug_feature_extraction(data_file, video_base_dir):
        print("⚠️ 特征提取存在问题，但继续调试...")
    '''
    # 4. 调试数据集创建
    dataset, output_dir = debug_dataset_creation()
    if dataset is None:
        print("⛔ 数据集创建失败，停止后续调试")
        return
    print("\n" + "=" * 80)
    print("🎉 调试完成!")

if __name__ == "__main__":
    main()