import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
from retinaface import RetinaFace
import cv2
import torch
import torchaudio
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Normalize


# 配置参数
TEXT_EMBEDDING_DIM = 768
AUDIO_FEATURE_SIZE = 40
VISUAL_FEATURE_SIZE = 35
SEED = 42

def extract_features(video_dir, csv_path, audio_dir=None):
    """
    从视频目录中提取特征并返回数据字典。
    """
    processor = MOSEIExtractor(language="unknown")
    label_mapping = {
        "SNEG": 0,
        "WNEG": 1,
        "NEUT": 2,
        "WPOS": 3,
        "SPOS": 4
    }

    # 初始化数据结构
    data = {
        "text": [],
        "language": [],
        "raw_text": [],
        "audio": [],
        "vision": [],
        "labels": [],
        "id": [],
        "class_labels": []
    }

    # 加载 CSV 文件
    with open(csv_path, "r") as f:
        for line in f.readlines()[1:]:  # 跳过表头
            video_id, clip_id, class_label = line.strip().split(",")
            video_path = Path(video_dir) / video_id / f"{clip_id}.mp4"

            if not video_path.exists():
                print(f"⚠️ 视频文件不存在: {video_path}")
                continue

            # 查找对应的音频文件
            audio_path = None
            if audio_dir:
                audio_path = processor.find_audio_file(video_path, audio_dir)

            # 提取特征
            language, raw_text, text_features = processor.extract_text_features(video_path, audio_path)
            audio_features = processor.extract_audio_features(video_path, audio_path)
            visual_features = processor.extract_visual_features(video_path)
            unique_id = f"{video_path.parent.name}_{video_path.stem}"

            # 映射 class_label
            mapped_class_label = label_mapping.get(class_label, -1)
            if mapped_class_label == -1:
                print(f"⚠️ 未知的 class_label: {class_label}")
                continue

            # 保存特征
            data["text"].append(text_features)
            data["language"].append(language)
            data["raw_text"].append(raw_text)
            data["audio"].append(audio_features)
            data["vision"].append(visual_features)
            data["id"].append(unique_id)
            data["class_labels"].append(mapped_class_label)
    return data


class MOSEIExtractor:
    def __init__(self, language="unknown"):
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化 Whisper 模型
        self.whisper_model = whisper.load_model("base").to(self.device)

        # 初始化 BERT 模型
         # ===== 加载英文 BERT =====
        self.en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.en_bert = BertModel.from_pretrained("bert-base-uncased").to(self.device).eval()

        # ===== 加载中文 BERT =====
        self.zh_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.zh_bert = BertModel.from_pretrained("bert-base-chinese").to(self.device).eval()

        # 初始化 ResNet 模型
        self.resnet_model = resnet18(pretrained=True)
        self.resnet_model = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])  # 去掉最后的分类层
        self.resnet_model = self.resnet_model.to(self.device).eval()

        # 图像预处理
        self.transform = Compose([
            ToPILImage(),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 初始化 PCA
        self.pca = PCA(n_components=VISUAL_FEATURE_SIZE)
        self.pca_fitted = False


    def find_audio_file(self, video_path, audio_dir):
        """根据视频文件名在音频文件夹中查找对应的音频文件"""
        video_name = Path(video_path).stem  # 获取视频文件名（不带扩展名）
        audio_path = Path(audio_dir) / f"{video_name}.wav"  # 假设音频文件是 .wav 格式
        if audio_path.exists():
            return audio_path
        else:
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
        video = VideoFileClip(video_path)
        if audio_path:
            temp_audio = audio_path
        else:
            # 从视频中提取音频
            temp_audio = "temp_audio.wav"
            video.audio.write_audiofile(temp_audio, logger=None)

        # 如果语言未知，先检测语言
        if self.language == "unknown":
            self.language = self.detect_language(temp_audio)

        # 使用 Whisper 提取文本
        result = self.whisper_model.transcribe(str(temp_audio), language=self.language, word_timestamps=True)
        if not audio_path:  # 如果是临时音频文件，删除它
            os.remove(temp_audio)

        # 拼接文本
        text_str = " ".join([segment["text"] for segment in result["segments"]])

        if self.language.startswith("zh"):  # 中文
            tokenizer = self.zh_tokenizer
            bert_model = self.zh_bert
        else:  # 英文默认
            tokenizer = self.en_tokenizer
            bert_model = self.en_bert
        # 使用 BERT 提取嵌入
        inputs = tokenizer(text_str, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return self.language, text_str, embedding[0]  # 返回 1D 向量

#-------------------------------------音频特征提取------------------------------------------------------
    def extract_audio_features(self, video_path, audio_path=None):
        """从视频或音频中提取音频特征并生成高级特征"""
        # 如果提供了音频路径，则直接使用音频文件
        video = VideoFileClip(video_path)
        if audio_path:
            temp_audio = audio_path
        else:
            # 从视频中提取音频
            temp_audio = "temp_audio.wav"
            video.audio.write_audiofile(temp_audio, logger=None)

        waveform, sr = torchaudio.load(temp_audio)
        if not audio_path:  # 如果是临时音频文件，删除它
            os.remove(temp_audio)

        # 提取音频帧特征
        mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40)(waveform)
        mfcc_mean = mfcc.mean(dim=-1).squeeze().numpy()
        return mfcc_mean
    
#-------------------------------------视频特征提取------------------------------------------------------
    def extract_visual_features(self, video_path):
        """从视频中提取视觉特征并生成高级特征"""
        cap = cv2.VideoCapture(video_path)
        features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            face_region = self._detect_face(frame)
            if face_region is not None:
                feature = self._extract_frame_features(face_region)
                features.append(feature)

        cap.release()

        if not features:
            return np.zeros(VISUAL_FEATURE_SIZE)

        features = np.mean(features, axis=0)
        if self.pca_fitted:
            features = self.pca.transform([features])[0]
        return features

    def _detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            return frame[y:y+h, x:x+w]
        return None

    def _extract_frame_features(self, face_region):
        face_tensor = self.transform(face_region).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.resnet_model(face_tensor).squeeze().cpu().numpy()
        return features

    def fit_pca(self, feature_list):
        if len(feature_list) == 0:
            raise ValueError("Feature list is empty. Cannot fit PCA.")
        self.pca.fit(feature_list)
        self.pca_fitted = True
        print("PCA model fitted successfully.")