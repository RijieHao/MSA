import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
import librosa
import cv2
import mediapipe as mp
import torch
from scipy.signal import find_peaks
from skimage.feature import hog

# 配置参数
TEXT_EMBEDDING_DIM = 768
AUDIO_FEATURE_SIZE = 40
VISUAL_FEATURE_SIZE = 46
SEED = 42

class MOSEIExtractor:
    def __init__(self, language="unknown"):
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化 Whisper 模型
        self.whisper_model = whisper.load_model("base")

        # 初始化 BERT 模型
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(self.device).eval()

        # 初始化 MediaPipe 面部检测
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )

    def find_audio_file(self, video_path, audio_dir):
        """根据视频文件名在音频文件夹中查找对应的音频文件"""
        video_name = Path(video_path).stem  # 获取视频文件名（不带扩展名）
        audio_path = Path(audio_dir) / f"{video_name}.wav"  # 假设音频文件是 .wav 格式
        if audio_path.exists():
            return audio_path
        else:
            print(f"⚠️ 未找到对应的音频文件: {audio_path}")
            return None

    #检测语言
    def detect_language(self, audio_path):
        """使用 Whisper 模型检测音频语言"""
        result = self.whisper_model.transcribe(audio_path, task="lang")
        detected_language = result["language"]
        print(f"Detected language: {detected_language}")
        return detected_language


#-------------------------------------文本特征提取------------------------------------------------------
    def extract_text_features(self, video_path, audio_path=None):
        """从视频或音频中提取文本特征并生成 BERT 嵌入"""
        # 如果提供了音频路径，则直接使用音频文件
        if audio_path:
            temp_audio = audio_path
        else:
            # 从视频中提取音频
            video = VideoFileClip(video_path)
            temp_audio = "temp_audio.wav"
            video.audio.write_audiofile(temp_audio, logger=None)

        # 如果语言未知，先检测语言
        if self.language == "unknown":
            self.language = self.detect_language(temp_audio)

        # 使用 Whisper 提取文本
        result = self.whisper_model.transcribe(temp_audio, language=self.language, word_timestamps=True)
        if not audio_path:  # 如果是临时音频文件，删除它
            os.remove(temp_audio)

        # 拼接文本
        words = [word["word"] for segment in result["segments"] for word in segment["words"]]
        text_str = " ".join(words)

        # 使用 BERT 提取嵌入
        inputs = self.tokenizer(text_str, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return self.language, text_str, embedding[0]  # 返回 1D 向量

#-------------------------------------音频特征提取------------------------------------------------------
    def extract_audio_features(self, video_path, audio_path=None):
        """从视频或音频中提取音频特征并生成高级特征"""
        # 如果提供了音频路径，则直接使用音频文件
        if audio_path:
            temp_audio = audio_path
        else:
            # 从视频中提取音频
            video = VideoFileClip(video_path)
            temp_audio = "temp_audio.wav"
            video.audio.write_audiofile(temp_audio, logger=None)

        y, sr = librosa.load(temp_audio, sr=22050)
        if not audio_path:  # 如果是临时音频文件，删除它
            os.remove(temp_audio)

        # 提取音频帧特征
        hop_length = max(1, len(y) // int(video.duration * video.fps))
        features = []
        for i in range(0, len(y), hop_length):
            frame = y[i:i + hop_length]
            if len(frame) > 0:
                frame_features = self._extract_covarep_frame_features(frame, sr)
                features.append(frame_features)
        features = np.array(features)

        # 生成高级特征
        return np.mean(features, axis=0) if features.size > 0 else np.zeros(AUDIO_FEATURE_SIZE)
    def _extract_covarep_frame_features(self, y_segment, sr):
        """提取单帧音频特征"""
        features = []
        # 12 个 MFCC 系数
        mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=12)
        features.extend(np.mean(mfcc, axis=1) if mfcc.shape[1] > 0 else np.zeros(12))

        # 基频特征 8
        f0 = librosa.yin(y_segment, fmin=85, fmax=400, sr=sr)
        f0_clean = f0[f0 > 0]
        pitch_features = [
            np.mean(f0_clean) if len(f0_clean) > 0 else 0.0,
            np.std(f0_clean) if len(f0_clean) > 0 else 0.0,
            np.median(f0_clean) if len(f0_clean) > 0 else 0.0,
            np.percentile(f0_clean, 25) if len(f0_clean) > 0 else 0.0,
            np.percentile(f0_clean, 75) if len(f0_clean) > 0 else 0.0,
            np.min(f0_clean) if len(f0_clean) > 0 else 0.0,
            np.max(f0_clean) if len(f0_clean) > 0 else 0.0,
            len(f0_clean) / len(f0) if len(f0) > 0 else 0.0
        ]
        features.extend(pitch_features)

        # 浊音/清音特征 6
        # 零交叉率
        zcr = np.mean(librosa.feature.zero_crossing_rate(y_segment))
        # 频谱重心
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_segment, sr=sr))
        # 频谱带宽
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr))
        # 频谱平坦度 - 修复API调用
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y_segment))
        # 能量
        energy = np.sum(y_segment ** 2) / len(y_segment)
        harmonic, percussive = librosa.effects.hpss(y_segment)
        hnr = np.sum(harmonic ** 2) / (np.sum(percussive ** 2) + 1e-8)
        features.extend([
            zcr, spectral_centroid, spectral_bandwidth, 
            spectral_flatness, energy, hnr
        ])


        # 5. 峰值斜率参数 (Peak Slope Parameters) - 4维
        hop_length = min(512, len(y_segment)//4)
        stft = librosa.stft(y_segment, hop_length=hop_length)
        envelope = np.abs(stft)
        envelope_mean = np.mean(envelope, axis=0)
        if len(envelope_mean) > 2:
            envelope_diff = np.diff(envelope_mean)
            slope_features = [
                np.mean(envelope_diff[envelope_diff > 0]) if np.any(envelope_diff > 0) else 0.0,
                np.mean(envelope_diff[envelope_diff < 0]) if np.any(envelope_diff < 0) else 0.0,
                np.std(envelope_diff),
                np.max(np.abs(envelope_diff))
            ]
            slope_features = [x if not np.isnan(x) else 0.0 for x in slope_features]
        else:
            slope_features = [0.0] * 4    
        features.extend(slope_features)

        # 6. 最大分散商 (Maxima Dispersion Quotients) - 4维
        peaks, _ = find_peaks(np.abs(y_segment), height=0.01)
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            dispersion_features = [
                np.mean(peak_intervals),
                np.std(peak_intervals),
                np.min(peak_intervals),
                np.max(peak_intervals)
            ]
        else:
            dispersion_features = [0.0] * 4
        features.extend(dispersion_features)

        # 其他情感特征补齐到 40 维
        remaining_dims = AUDIO_FEATURE_SIZE - len(features)
        chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1) if chroma.shape[1] > 0 else np.zeros(12)
        additional_features = list(chroma_mean[:remaining_dims])
        features.extend(additional_features)

        features = features[:AUDIO_FEATURE_SIZE]
        while len(features) < AUDIO_FEATURE_SIZE:
            features.append(0.0)
        return np.array(features)
    

#-------------------------------------视频特征提取------------------------------------------------------
    def extract_visual_features(self, video_path):
        """从视频中提取视觉特征并生成高级特征"""
        cap = cv2.VideoCapture(video_path)
        features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_results = self.face_detection.process(rgb_frame)
            mesh_results = self.face_mesh.process(rgb_frame)

            if mesh_results.multi_face_landmarks and detection_results.detections:
                landmarks = mesh_results.multi_face_landmarks[0]
                features.append(self._extract_visual_frame_features(landmarks, frame))
            else:
                features.append(np.zeros(VISUAL_FEATURE_SIZE))

        cap.release()
        features = np.array(features)

        # 生成高级特征
        return np.mean(features, axis=0) if features.size > 0 else np.zeros(VISUAL_FEATURE_SIZE)

    def _extract_visual_frame_features(self, landmarks, frame):
        """提取单帧视觉特征"""
        features = []
        # 10 点人脸关键点特征 20
        key_indices = [1, 33, 263, 61, 291, 199, 234, 454, 10, 152]
        for idx in key_indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y])
            else:
                features.extend([0.0, 0.0])

        # HoG 特征 3
        h, w = frame.shape[:2]
        # 获取面部边界框
        x_coords = [landmarks.landmark[i].x * w for i in range(len(landmarks.landmark))]
        y_coords = [landmarks.landmark[i].y * h for i in range(len(landmarks.landmark))]
        x1, x2 = int(min(x_coords)), int(max(x_coords))
        y1, y2 = int(min(y_coords)), int(max(y_coords))
        # 添加边距
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        face_region = frame[y1:y2, x1:x2]
        if face_region.size > 0:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (64, 64))
            hog_features = hog(
                gray_face,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                feature_vector=True
            )
            features.extend(hog_features[:3])
        else:
            features.extend([0.0, 0.0, 0.0])

        # 头部姿态 # 简化为 3 维
        nose_tip = landmarks.landmark[1]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[362]
        # Roll (头部倾斜)
        eye_angle = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
        roll = np.degrees(eye_angle)
        # Pitch (俯仰)
        pitch = (nose_tip.y - 0.5) * 60
        # Yaw (偏航)
        face_center_x = (left_eye.x + right_eye.x) / 2
        yaw = (nose_tip.x - face_center_x) * 120
        head_pose = [roll, pitch, yaw]
        features.extend(head_pose)  
        

        # 眼部注视方向 4 维
        left_eye_center = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                                  for i in [33, 133]], axis=0)
        right_eye_center = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                                   for i in [362, 263]], axis=0)
        # 计算注视方向
        left_gaze_x = (left_eye_center[0] - 0.3) * 2
        left_gaze_y = (left_eye_center[1] - 0.4) * 2
        right_gaze_x = (right_eye_center[0] - 0.7) * 2
        right_gaze_y = (right_eye_center[1] - 0.4) * 2
        features.extend([left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y])

        # FACS 动作单元# 简化为 10 维
        aus = []
        # AU1 - 内眉上抬
        au1 = max(0, 0.5 - landmarks.landmark[55].y) * 10
        aus.append(au1)
        # AU2 - 外眉上抬
        au2 = max(0, 0.4 - landmarks.landmark[70].y) * 10
        aus.append(au2)
        # AU4 - 眉头紧锁
        brow_distance = abs(landmarks.landmark[55].x - landmarks.landmark[70].x)
        au4 = max(0, 0.1 - brow_distance) * 50
        aus.append(au4)
        # AU5 - 上眼睑上抬
        left_eye_open = abs(landmarks.landmark[33].y - landmarks.landmark[145].y)
        au5 = left_eye_open * 20
        aus.append(au5)
        # AU6 - 脸颊上抬
        cheek_height = landmarks.landmark[116].y
        au6 = max(0, 0.6 - cheek_height) * 15
        aus.append(au6)
        # AU9 - 鼻皱
        nose_width = abs(landmarks.landmark[125].x - landmarks.landmark[141].x)
        au9 = max(0, nose_width - 0.02) * 100
        aus.append(au9)
        # AU12 - 嘴角上扬
        mouth_corner_avg = (landmarks.landmark[61].y + landmarks.landmark[84].y) / 2
        mouth_center = landmarks.landmark[13].y
        au12 = max(0, mouth_center - mouth_corner_avg) * 50
        aus.append(au12)
        # AU15 - 嘴角下拉
        au15 = max(0, mouth_corner_avg - mouth_center) * 50
        aus.append(au15)
        # AU20 - 嘴唇水平拉伸
        mouth_width = abs(landmarks.landmark[61].x - landmarks.landmark[291].x)
        au20 = mouth_width * 100
        aus.append(au20)
        # AU25 - 嘴唇分离
        mouth_open = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
        au25 = mouth_open * 100
        aus.append(au25)
        features.extend(aus) 


        #脸部情感特征 6维
        emotions = []
        # Happiness - 基于嘴角上扬
        left_corner = landmarks.landmark[61].y
        right_corner = landmarks.landmark[84].y
        mouth_center = landmarks.landmark[13].y
        corner_avg = (left_corner + right_corner) / 2
        mouth_corner_lift = mouth_center - corner_avg
        happiness = max(0, mouth_corner_lift) * 5
        emotions.append(happiness)
        
        # Sadness - 基于嘴角下拉和眉毛下垂
        mouth_corner_drop = -min(0, mouth_corner_lift)
        brow_drop = max(0, 0.45 - landmarks.landmark[55].y)
        sadness = (mouth_corner_drop + brow_drop) * 3
        emotions.append(sadness)
        
        # Anger - 基于眉毛紧锁
        brow_furrow = max(0, 0.1 - abs(landmarks.landmark[55].x - landmarks.landmark[70].x))
        anger = brow_furrow * 20
        emotions.append(anger)
        
        # Disgust - 基于鼻子皱起
        nose_scrunch = abs(landmarks.landmark[125].x - landmarks.landmark[141].x)
        disgust = max(0, nose_scrunch - 0.02) * 50
        emotions.append(disgust)
        
        # Surprise - 基于眉毛上抬和嘴巴张开
        brow_raise = max(0, 0.4 - landmarks.landmark[70].y)
        mouth_open = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
        surprise = (brow_raise + mouth_open) * 10
        emotions.append(surprise)
        
        # Fear - 基于眼睛张大和眉毛上抬
        eye_wide = abs(landmarks.landmark[33].y - landmarks.landmark[145].y)
        fear = (eye_wide + brow_raise) * 8
        emotions.append(fear) 
        features.extend(emotions) 

        features = features[:VISUAL_FEATURE_SIZE]
        while len(features) < VISUAL_FEATURE_SIZE:
            features.append(0.0)
        return np.array(features)


#-------------------------------------生成pkl文件------------------------------------------------------
    def process_dataset(self, video_dir, csv_path, output_dir, audio_dir=None):
        """处理整个数据集并生成 .pkl 文件"""
        # 加载标签和数据集划分信息
        splits = {"train": [], "valid": [], "test": []}
        labels = {}

        with open(csv_path, "r") as f:
            for line in f.readlines()[1:]:  # 跳过表头
                video_id, clip_id, label, split = line.strip().split(",")
                video_path = Path(video_dir) / video_id / f"{clip_id}.mp4"
                splits[split].append((video_path, float(label)))

        # 初始化数据结构
        data = {split: {"text": [], "language": [], "raw_text": [], "audio": [], "vision": [], "labels": [], "id": []} for split in splits}

        # 提取特征
        for split, items in splits.items():
            for video_path, label in tqdm(items, desc=f"Processing {split} set"):
                if not video_path.exists():
                    print(f"⚠️ 视频文件不存在: {video_path}")
                    continue

                # 查找对应的音频文件
                audio_path = None
                if audio_dir:
                    audio_path = self.find_audio_file(video_path, audio_dir)

                # 提取特征
                language, raw_text, text_features = self.extract_text_features(video_path, audio_path)
                audio_features = self.extract_audio_features(video_path, audio_path)
                visual_features = self.extract_visual_features(video_path)
                unique_id = f"{video_path.parent.name}_{video_path.stem}"

                # 保存特征
                data[split]["text"].append(text_features)
                data[split]["language"].append(language)
                data[split]["raw_text"].append(raw_text)
                data[split]["audio"].append(audio_features)
                data[split]["vision"].append(visual_features)
                data[split]["labels"].append(label)
                data[split]["id"].append(unique_id)  # 使用唯一 ID


        # 保存 .pkl 文件
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            with open(output_dir / f"{split}_data.pkl", "wb") as f:
                pickle.dump(data[split], f)

        # 保存 metadata
        metadata = {
            "text_dim": TEXT_EMBEDDING_DIM,
            "audio_dim": AUDIO_FEATURE_SIZE,
            "visual_dim": VISUAL_FEATURE_SIZE,
            "num_classes": 1,
            "train_samples": len(data["train"]["labels"]),
            "val_samples": len(data["valid"]["labels"]),
            "test_samples": len(data["test"]["labels"]),
        }
        with open(output_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print("Processing completed and files saved successfully.")
#-------------------------------------运行主函数-----------------------------------------------------
if __name__ == "__main__":
    processor = MOSEIExtractor(language="zh")
    processor.process_dataset(
        video_dir="video2pkl/video2pkl/ch_video",
        csv_path="video2pkl/video2pkl/ch_video.csv",
        output_dir="E:/kaggle/MSAbypkl/data_pkl/ch_pkl",
        audio_dir=None
    )