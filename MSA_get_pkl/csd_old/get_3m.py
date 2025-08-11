import os
import pickle
import cv2
import librosa
import numpy as np
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import mediapipe as mp
from pathlib import Path
import pickle
import h5py
import pandas as pd
from tqdm import tqdm
import gensim.downloader as api
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import warnings
from skimage.feature import hog
import jieba
import fasttext
import fasttext.util

class AccurateMOSEIExtractor:
    """
    更准确还原MOSEI数据集特征提取的类
    基于论文描述实现原始特征提取方法
    支持中英文
    """
    
    def __init__(self, language="en"):
        self.language = language
        
        # 加载Whisper用于语音识别和对齐
        self.whisper_model = whisper.load_model("base")
        
        # 根据语言选择词嵌入模型
        if language in ["zh", "chinese", "中文"]:
            print("Initializing Chinese word embeddings...")
            self._load_chinese_embeddings()
        else:
            print("Initializing English GloVe embeddings...")
            self._load_english_embeddings()

        # 初始化MTCNN替代品 (MediaPipe)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        # 初始化面部关键点检测 (OpenFace替代)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print(f"Initialized AccurateMOSEIExtractor for language: {language}")
    
    def _load_chinese_embeddings(self):
        """加载中文词嵌入模型（对标GloVe）"""
        try:
            # 方案1: 尝试加载FastText中文模型
            self._load_fasttext_chinese()
        except Exception as e:
            print(f"Failed to load FastText: {e}")
            try:
                # 方案2: 尝试加载预训练中文Word2Vec
                self._load_chinese_word2vec()
            except Exception as e2:
                print(f"Failed to load Word2Vec: {e2}")

    def _load_fasttext_chinese(self):
        """使用FastText中文模型"""
        try:
            print("Loading FastText Chinese model...")
            # 下载中文FastText模型
            fasttext.util.download_model('zh', if_exists='ignore')
            self.word_model = fasttext.load_model('cc.zh.300.bin')
            self.word_embedding_dim = 300
            self.embedding_type = "fasttext"
            print("✓ Loaded FastText Chinese: 300d")
            
        except Exception as e:
            raise e

    def _load_chinese_word2vec(self):
        """加载预训练中文Word2Vec模型"""
        try:
            from gensim.models import KeyedVectors
            
            # 可能的中文Word2Vec模型路径
            model_paths = [
                "data/embeddings/chinese_word2vec_300d.txt",
                "data/embeddings/sgns.weibo.bigram-char",
                "data/embeddings/tencent-ailab-embedding-zh-d200-v0.2.0.txt"
            ]
            
            for model_path in model_paths:
                if Path(model_path).exists():
                    print(f"Loading Chinese Word2Vec from {model_path}...")
                    self.word_model = KeyedVectors.load_word2vec_format(
                        model_path, binary=False, unicode_errors='ignore'
                    )
                    self.word_embedding_dim = self.word_model.vector_size
                    self.embedding_type = "word2vec"
                    print(f"✓ Loaded Chinese Word2Vec: {self.word_embedding_dim}d")
                    return
            
            raise FileNotFoundError("No Chinese Word2Vec model found")
            
        except Exception as e:
            raise e

    def _load_english_embeddings(self):
        """加载英文GloVe嵌入"""
        try:
            import gensim.downloader as api
            print("Loading GloVe word embeddings...")
            self.word_model = api.load("glove-wiki-gigaword-300")
            self.word_embedding_dim = 300
            self.embedding_type = "glove"
            print("✓ Loaded GloVe: 300d")
        except Exception as e:
            print(f"Failed to load GloVe: {e}")
            self.word_model = None
            self.word_embedding_dim = 300
            self.embedding_type = "random"

    def extract_word_features(self, video_path):
        """
        提取基于词嵌入的特征 - 中英文统一接口
        
        Returns:
            list: [[word_embedding, start_time, end_time], ...] 格式
        """
        # 1. 从视频提取音频
        video = VideoFileClip(video_path)
        temp_audio = f"temp_audio_for_{self.language}.wav"
        video.audio.write_audiofile(temp_audio, logger=None)
        
        # 2. 使用Whisper进行词级对齐
        whisper_language = "zh" if self.language in ["zh", "chinese", "中文"] else self.language
        result = self.whisper_model.transcribe(
            temp_audio, 
            language=whisper_language,
            word_timestamps=True
        )
        
        # 3. 转换为词嵌入序列
        word_features = []
        
        if 'segments' in result:
            for segment in result['segments']:
                if 'words' in segment:
                    for word_info in segment['words']:
                        word = word_info.get('word', '').strip()
                        start = word_info.get('start', 0.0)
                        end = word_info.get('end', 0.0)
                        
                        if word:
                            if self.language in ["zh", "chinese", "中文"]:
                                # 中文分词处理
                                segmented_words = self._chinese_word_segmentation(word)
                                for seg_word in segmented_words:
                                    if seg_word.strip():
                                        word_embedding = self._get_chinese_word_embedding(seg_word)
                                        word_features.append([word_embedding, start, end])
                            else:
                                # 英文处理
                                word_clean = word.lower().strip()
                                word_embedding = self._get_english_word_embedding(word_clean)
                                word_features.append([word_embedding, start, end])
        
        # 4. 清理临时文件
        os.remove(temp_audio)
        video.close()
        
        return word_features

    def _chinese_word_segmentation(self, text):
        """中文分词"""
        try:
            words = list(jieba.cut(text, cut_all=False))
            return [w.strip() for w in words if w.strip()]
        except ImportError:
            print("Warning: jieba not available, using character-level segmentation")
            # 如果jieba不可用，按字符分割
            return list(text)

    def _get_chinese_word_embedding(self, word):
        """获取中文词嵌入"""
        if self.embedding_type == "fasttext":
            return self.word_model.get_word_vector(word).astype(np.float32)
        elif self.embedding_type == "word2vec":
            if word in self.word_model:
                return self.word_model[word].astype(np.float32)
            else:
                return np.zeros(self.word_embedding_dim, dtype=np.float32)
        elif self.embedding_type == "char":
            return self._get_character_embedding(word)
        else:
            return np.zeros(self.word_embedding_dim, dtype=np.float32)

    def _get_character_embedding(self, word):
        """获取词的字符级平均嵌入"""
        char_embeddings = []
        
        for char in word:
            if char in self.char_vocab:
                char_idx = self.char_vocab[char]
            else:
                char_idx = self.char_vocab['<UNK>']
            
            char_embeddings.append(self.char_embeddings[char_idx])
        
        if char_embeddings:
            # 返回字符嵌入的平均值
            return np.mean(char_embeddings, axis=0).astype(np.float32)
        else:
            return np.zeros(self.word_embedding_dim, dtype=np.float32)

    def _get_english_word_embedding(self, word):
        """获取英文词嵌入（GloVe）"""
        if self.embedding_type == "glove" and self.word_model and word in self.word_model:
            return self.word_model[word].astype(np.float32)
        else:
            # 如果词汇不在GloVe中，返回零向量
            return np.zeros(self.word_embedding_dim, dtype=np.float32)

    # 保持原有的音频和视觉特征提取方法不变
    def extract_covarep_acoustic_features(self, video_path):
        """
        提取COVAREP风格的声学特征 - 完全修复版本
        
        Returns:
            np.ndarray: shape [时间步数,40] 的COVAREP兼容特征
        """
        try:
            print(f"🔊 提取音频特征: {video_path}")
            
            # 1. 从视频提取音频
            video = VideoFileClip(video_path)
            frame_rate = video.fps  # 获取实际帧率
            if video.audio is None:
                print(f"⚠️ 视频没有音频轨道")
                video.close()
                return np.zeros((1, 40))
            
            temp_audio = "temp_audio_for_covarep.wav"
            
            # 强制使用 16kHz 避免采样率问题
            video.audio.write_audiofile(temp_audio, 
                                       fps=22050,  # 强制16kHz
                                       logger=None,
                                       )
            video.close()
            
            # 2. 使用librosa加载音频，确保采样率
            y, sr = librosa.load(temp_audio, sr=22050)  # 强制16kHz
            
            print(f"🎵 音频信息: {len(y)/sr:.2f}秒, 采样率: {sr}Hz")
            
            # 3. 计算帧数 (与视频对齐)
            video_duration = len(y) / sr
            n_frames = max(1, int(video_duration * frame_rate))
            
            # 4. 提取特征
            features_sequence = []
            hop_length = max(1, len(y) // n_frames)
            
            for i in range(n_frames):
                start_idx = i * hop_length
                end_idx = min((i + 1) * hop_length, len(y))
                
                if start_idx >= len(y):
                    break
                    
                y_segment = y[start_idx:end_idx]
                
                if len(y_segment) == 0:
                    frame_features = np.zeros(40)
                else:
                    frame_features = self._extract_covarep_frame_features(y_segment, sr)
                
                features_sequence.append(frame_features)
            
            # 5. 清理临时文件
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            acoustic_features = np.array(features_sequence)
            print(f"✅ 声学特征形状: {acoustic_features.shape}")
            return acoustic_features
            
        except Exception as e:
            print(f"❌ 声学特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 清理临时文件
            temp_audio = "temp_audio_for_covarep.wav"
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return np.zeros((1, 40))
    
    def _extract_covarep_frame_features(self, y_segment, sr):
        """按COVAREP标准提取单帧40维特征 - 修复版本"""
        features = []
        fmin = 85
        frame_length = min(369, len(y_segment) // 2)   
        try:
            # 1. 12个MFCC系数
            if len(y_segment) > 512:
                mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=12)
                mfcc_mean = np.mean(mfcc, axis=1) if mfcc.shape[1] > 0 else np.zeros(12)
                features.extend(mfcc_mean)
            else:
                features.extend([0.0] * 12)
            
            # 2. 基频特征 (Pitch) - 8维 - 完全修复
            try:
                if len(y_segment) > frame_length and frame_length > 0:
                    f0 = librosa.yin(y_segment, 
                                   fmin=fmin, 
                                   fmax=min(400, sr//4),
                                   frame_length=frame_length)
                    
                    if len(f0) > 0:
                        f0_clean = f0[f0 > 0]
                        if len(f0_clean) > 0:
                            pitch_features = [
                                np.mean(f0_clean),
                                np.std(f0_clean),
                                np.median(f0_clean),
                                np.percentile(f0_clean, 25),
                                np.percentile(f0_clean, 75),
                                np.min(f0_clean),
                                np.max(f0_clean),
                                len(f0_clean) / len(f0)
                            ]
                        else:
                            pitch_features = [0.0] * 8
                    else:
                        pitch_features = [0.0] * 8
                else:
                    pitch_features = [0.0] * 8
                
            except Exception as pitch_error:
                print(f"⚠️ Pitch extraction failed: {pitch_error}")
                pitch_features = [0.0] * 8
                
            features.extend(pitch_features)
            
            # 3. 浊音/清音分割特征 (Voiced/Unvoiced) - 6维 - 修复API调用
            if len(y_segment) > 512:
                try:
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
                    
                    # 谐波噪声比
                    try:
                        harmonic, percussive = librosa.effects.hpss(y_segment)
                        hnr = np.sum(harmonic ** 2) / (np.sum(percussive ** 2) + 1e-8)
                    except:
                        hnr = 1.0  # 默认值
                    
                    voiced_unvoiced_features = [zcr, spectral_centroid, spectral_bandwidth, 
                                              spectral_flatness, energy, hnr]
                except Exception as vu_error:
                    print(f"⚠️ Voiced/Unvoiced features failed: {vu_error}")
                    voiced_unvoiced_features = [0.0] * 6
            else:
                voiced_unvoiced_features = [0.0] * 6
            features.extend(voiced_unvoiced_features)
        
            
            # 5. 峰值斜率参数 (Peak Slope Parameters) - 4维
            if len(y_segment) > 256:
                try:
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
                except Exception as slope_error:
                    print(f"⚠️ Slope features failed: {slope_error}")
                    slope_features = [0.0] * 4
            else:
                slope_features = [0.0] * 4
            features.extend(slope_features)
            
            # 6. 最大分散商 (Maxima Dispersion Quotients) - 4维
            if len(y_segment) > 512:
                try:
                    from scipy.signal import find_peaks
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
                except Exception as disp_error:
                    print(f"⚠️ Dispersion features failed: {disp_error}")
                    dispersion_features = [0.0] * 4
            else:
                dispersion_features = [0.0] * 4
            features.extend(dispersion_features)
            
            # 7. 其他情感相关特征补齐40维
            remaining_dims = 40 - len(features)
            if remaining_dims > 0:
                try:
                    if len(y_segment) > 512:
                        hop_length = min(512, len(y_segment)//4)
                        chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr, hop_length=hop_length)
                        chroma_mean = np.mean(chroma, axis=1) if chroma.shape[1] > 0 else np.zeros(12)
                        additional_features = list(chroma_mean[:remaining_dims])
                    else:
                        additional_features = [0.0] * remaining_dims
                except Exception as chroma_error:
                    print(f"⚠️ Chroma features failed: {chroma_error}")
                    additional_features = [0.0] * remaining_dims
                features.extend(additional_features)
            
            # 确保正好40维
            features = features[:40]
            while len(features) < 40:
                features.append(0.0)
            
        except Exception as e:
            print(f"❌ Error extracting COVAREP features: {e}")
            features = [0.0] * 40
        
        return np.array(features, dtype=np.float32)
    
    def extract_openface_visual_features(self, video_path):
        """
        提取OpenFace风格的视觉特征 - 还原原始MOSEI视觉特征
        包含68个面部关键点、20个形状参数、HoG特征、头部姿态、眼部注视
        
        Returns:
            np.ndarray: shape [时间步数, 709] 的OpenFace兼容特征
        """
        cap = cv2.VideoCapture(video_path)
        features_sequence = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            # MTCNN替代: 使用MediaPipe进行人脸检测
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_results = self.face_detection.process(rgb_frame)
            
            # 面部关键点检测
            mesh_results = self.face_mesh.process(rgb_frame)
            
            if mesh_results.multi_face_landmarks and detection_results.detections:
                landmarks = mesh_results.multi_face_landmarks[0]
                detection = detection_results.detections[0]
                frame_features = self._extract_openface_frame_features(
                    landmarks, detection, frame
                )
            else:
                # 没有检测到人脸时使用零特征
                frame_features = np.zeros(35)
            
            features_sequence.append(frame_features)
        
        cap.release()
        
        if len(features_sequence) == 0:
            features_sequence = [np.zeros(35)]
        
        return np.array(features_sequence)
    

    '''
    下面是709维脸部特征提取的内容的简化，注释前为709维
    注释后为35维
    '''
    def _extract_openface_frame_features(self, landmarks, detection, frame):
        """按OpenFace标准提取单帧709维特征"""
        features = []
        h, w = frame.shape[:2]
        
        # 1. 68个面部关键点坐标 (136维: 68点 × 2坐标)
        #landmark_coords = self._extract_68_facial_landmarks(landmarks)
        #features.extend(landmark_coords)  # 136维

        #1.10个关键点坐标（选取部分有代表性的点）(20维)
        key_indices = [1, 33, 263, 61, 291, 199, 234, 454, 10, 152]  # 鼻尖、左右眼、嘴角、脸颊等
        for idx in key_indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y])
            else:
                features.extend([0.0, 0.0])
        
        # 2. 20个面部形状参数 (PCA降维后的形状描述)
        #shape_params = self._extract_facial_shape_parameters(landmarks)
        #features.extend(shape_params)  # 20维
        
        # 3. 面部HoG特征 (简化版本3维)
        hog_features = self._extract_facial_hog_features(landmarks, frame)
        #features.extend(hog_features)  # 100维
        features.extend(hog_features[:3])  # 3维
        
        # 4. 头部姿态 (6维: roll, pitch, yaw, x, y, z)(简化3维)
        head_pose = self._extract_head_pose(landmarks, frame.shape)
        #features.extend(head_pose)  # 6维
        features.extend(head_pose[:3])  # 3维
        
        # 5. 眼部注视方向 (4维)
        eye_gaze = self._extract_eye_gaze(landmarks)
        features.extend(eye_gaze)  # 4维
        
        # 6. FACS动作单元 (17维AU强度)(简化5维)
        action_units = self._extract_facial_action_units(landmarks)
        #features.extend(action_units)  # 17维
        features.extend(action_units[:5])  # 5维
        
        # 7. 6种基本情感 (Emotient FACET风格)
        #basic_emotions = self._extract_basic_emotions(landmarks)
        #features.extend(basic_emotions)  # 6维
        
        # 8. 深度人脸嵌入特征 (简化版)
        #face_embeddings = self._extract_face_embeddings(landmarks, frame)
        #features.extend(face_embeddings)  # 128维
        
        # 9. 其他特征补齐到709维
        #remaining_dims = 709 - len(features)
        #if remaining_dims > 0:
        #    additional_features = self._extract_additional_visual_features(
        #        landmarks, frame, remaining_dims
        #    )
        #    features.extend(additional_features)
        
        # 确保正好709维
        #features = features[:709]
        features = features[:35]
        #while len(features) < 709:
        while len(features) < 35:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_68_facial_landmarks(self, landmarks):
        """提取68个面部关键点 (OpenFace标准)"""
        # MediaPipe有468个点，选择对应OpenFace的68个关键点
        openface_68_indices = [
            # 下巴轮廓 (17个点: 0-16)
            172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
            # 右眉毛 (5个点: 17-21)
            70, 63, 105, 66, 107,
            # 左眉毛 (5个点: 22-26)
            55, 65, 52, 53, 46,
            # 鼻子 (9个点: 27-35)
            168, 8, 9, 10, 151, 195, 197, 196, 3,
            # 右眼 (6个点: 36-41)
            33, 7, 163, 144, 145, 153,
            # 左眼 (6个点: 42-47)
            362, 382, 381, 380, 374, 373,
            # 嘴部外轮廓 (12个点: 48-59)
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            # 嘴部内轮廓 (8个点: 60-67)
            78, 95, 88, 178, 87, 14, 317, 402
        ]
        
        coords = []
        for i in range(68):
            if i < len(openface_68_indices):
                idx = openface_68_indices[i]
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    coords.extend([landmark.x, landmark.y])
                else:
                    coords.extend([0.0, 0.0])
            else:
                coords.extend([0.0, 0.0])
        
        return coords[:136]  # 确保136维
    
    def _extract_facial_shape_parameters(self, landmarks):
        """提取20个面部形状参数 (PCA降维)"""
        # 简化版本：基于关键点计算形状描述符
        params = []
        
        # 面部宽高比
        face_width = abs(landmarks.landmark[234].x - landmarks.landmark[454].x)
        face_height = abs(landmarks.landmark[10].y - landmarks.landmark[152].y)
        params.append(face_width / (face_height + 1e-8))
        
        # 眼睛宽度比例
        left_eye_width = abs(landmarks.landmark[33].x - landmarks.landmark[133].x)
        right_eye_width = abs(landmarks.landmark[362].x - landmarks.landmark[263].x)
        params.extend([left_eye_width, right_eye_width])
        
        # 嘴巴参数
        mouth_width = abs(landmarks.landmark[61].x - landmarks.landmark[291].x)
        mouth_height = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
        params.extend([mouth_width, mouth_height])
        
        # 鼻子参数
        nose_width = abs(landmarks.landmark[125].x - landmarks.landmark[141].x)
        nose_height = abs(landmarks.landmark[19].y - landmarks.landmark[1].y)
        params.extend([nose_width, nose_height])
        
        # 补齐到20维
        while len(params) < 20:
            params.append(0.0)
        
        return params[:20]
    
    def _extract_facial_hog_features(self, landmarks, frame):
        """提取面部HoG特征 (简化版本)"""
        try:
            
            # 提取面部区域
            face_region = self._extract_face_region(landmarks, frame)
            
            if face_region is not None:
                # 转换为灰度
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                # 调整大小
                gray_face = cv2.resize(gray_face, (64, 64))
                
                # 提取HoG特征
                hog_features = hog(
                    gray_face,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys',
                    feature_vector=True
                )
                
                # 降维到100维
                if len(hog_features) > 100:
                    hog_features = hog_features[:100]
                else:
                    hog_features = list(hog_features) + [0.0] * (100 - len(hog_features))
            else:
                hog_features = [0.0] * 100
                
        except Exception as e:
            hog_features = [0.0] * 100
        
        return hog_features
    
    def _extract_face_region(self, landmarks, frame):
        """从frame中提取面部区域"""
        try:
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
                return face_region
            else:
                return None
                
        except Exception as e:
            return None
    
    def _extract_head_pose(self, landmarks, frame_shape):
        """提取头部姿态 (6维)"""
        # 简化的头部姿态估计
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
        
        # 位置 (相对于图像中心)
        h, w = frame_shape[:2]
        x_pos = (nose_tip.x - 0.5) * w
        y_pos = (nose_tip.y - 0.5) * h
        z_pos = abs(landmarks.landmark[10].y - landmarks.landmark[152].y) * 100
        
        return [roll, pitch, yaw, x_pos, y_pos, z_pos]
    
    def _extract_eye_gaze(self, landmarks):
        """提取眼部注视方向 (4维)"""
        # 简化的注视方向估计
        left_eye_center = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                                  for i in [33, 133]], axis=0)
        right_eye_center = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                                   for i in [362, 263]], axis=0)
        
        # 计算注视方向
        left_gaze_x = (left_eye_center[0] - 0.3) * 2
        left_gaze_y = (left_eye_center[1] - 0.4) * 2
        right_gaze_x = (right_eye_center[0] - 0.7) * 2
        right_gaze_y = (right_eye_center[1] - 0.4) * 2
        
        return [left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y]
    
    def _extract_facial_action_units(self, landmarks):
        """提取FACS动作单元强度 (17维)"""
        # 基于面部关键点估计主要AU
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
        
        # 补齐到17维
        while len(aus) < 17:
            aus.append(0.0)
        
        return aus[:17]
    
    def _extract_basic_emotions(self, landmarks):
        """提取6种基本情感 (Emotient FACET风格)"""
        # 基于面部关键点的简化情感识别
        emotions = []
        
        # Happiness - 基于嘴角上扬
        mouth_corner_lift = self._calculate_mouth_corner_lift(landmarks)
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
        
        return emotions
    
    def _calculate_mouth_corner_lift(self, landmarks):
        """计算嘴角上扬程度"""
        left_corner = landmarks.landmark[61].y
        right_corner = landmarks.landmark[84].y
        mouth_center = landmarks.landmark[13].y
        
        corner_avg = (left_corner + right_corner) / 2
        return mouth_center - corner_avg
    
    def _extract_face_embeddings(self, landmarks, frame):
        """提取深度人脸嵌入 (简化版本)"""
        # 这里可以集成真正的人脸识别模型如FaceNet
        # 目前使用基于关键点的简化嵌入
        embedding = []
        
        # 基于关键点计算几何特征作为嵌入
        for i in range(0, min(128, len(landmarks.landmark))):
            if i < len(landmarks.landmark):
                landmark = landmarks.landmark[i]
                embedding.extend([landmark.x, landmark.y])
            else:
                embedding.extend([0.0, 0.0])
        
        # 调整到128维
        while len(embedding) < 128:
            embedding.append(0.0)
        
        return embedding[:128]
    
    def _extract_additional_visual_features(self, landmarks, frame, num_features):
        """提取额外的视觉特征以达到709维"""
        features = []
        
        # 添加更多几何特征
        for i in range(num_features):
            if i < len(landmarks.landmark):
                landmark = landmarks.landmark[i % len(landmarks.landmark)]
                features.append(landmark.x * landmark.y)  # 简单的组合特征
            else:
                features.append(0.0)
        
        return features[:num_features]
    
    def save_to_csd_format(self, data, output_path, description="", metadata={}):
        with h5py.File(output_path, 'w') as f:
            # 创建顶级组 computational_sequences
            top_group = f.create_group("computational_sequences")
            
            # 创建 data 组
            data_group = top_group.create_group("data")
            for segment_id, segment_data in data.items():
                features = np.array(segment_data["features"], dtype=np.float32)
                intervals = np.array(segment_data["intervals"], dtype=np.float32)
                seg_group = data_group.create_group(segment_id)
                seg_group.create_dataset("features", data=features, compression='gzip')
                seg_group.create_dataset("intervals", data=intervals, compression='gzip')
            
            # 创建 metadata 组
            metadata_group = top_group.create_group("metadata")
            for key, value in metadata.items():
                try:
                    if key == "md5" and value is None:
                        value = ""  # 或者使用 "None"
                    # 如果值是 None 或其他复杂类型，转换为字符串
                    elif value is None or isinstance(value, (list, tuple, dict)):
                        value = str(value)
                    metadata_group.attrs[key] = value
                except Exception as e:
                    print(f"❌ Error setting metadata attribute {key}: {e}")
            
            # 添加 description 属性
            metadata_group.attrs["description"] = description
        
        print(f"✅ CSD 文件保存成功: {output_path}")

    def process_labels_csv_to_csd(self, csv_path):
        """
        将您的meta.csv转换为MOSEI格式的labels数据
        
        Args:
            csv_path (str): meta.csv文件路径
            
        Returns:
            dict: labels数据字典
        """
        print(f"Processing labels from {csv_path}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        labels_data = {}
        
        for _, row in df.iterrows():
            try:
                # 根据您的CSV格式：
                # 第1列：video_id (文件夹名)
                # 第2列：clip_id (文件名)  
                # 第4列：label (整体情感标签)
                # 第5列：label_T (文本情感标签)
                # 第6列：label_A (音频情感标签)
                # 第7列：label_V (视频情感标签)
                
                video_id = str(row.iloc[0])     # video_0001
                clip_id = str(row.iloc[1])      # 0001
                overall_label = float(row.iloc[2])  # label
                
                # 使用video_id + clip_id作为segment_id，与视频文件对应
                segment_id = f"{video_id}_{clip_id}"
                
                # 创建MOSEI标签格式 - 使用整体标签
                labels_data[segment_id] = {
                    'features': np.array([[overall_label]], dtype=np.float32),
                    'intervals': np.array([[0, 1]], dtype=np.int32)
                }
                
            except Exception as e:
                print(f"Error processing label for row {_}: {e}")
                # 使用默认中性标签
                segment_id = f"error_{_}"
                labels_data[segment_id] = {
                    'features': np.array([[0.0]], dtype=np.float32),
                    'intervals': np.array([[0, 1]], dtype=np.int32)
                }
        
        print(f"✓ Converted {len(labels_data)} labels")
        return labels_data

    def process_video_to_accurate_mosei_format(self, video_path):
        """
        将MP4视频转换为更准确的MOSEI数据集格式
        
        Returns:
            dict: 包含三个模态的准确特征字典
        """
        print(f"Processing video with accurate MOSEI features: {video_path}")
        
        # 1. 提取词嵌入序列
        print("Extracting word-based language features...")
        word_features = self.extract_word_features(video_path)
        
        # 2. 提取COVAREP音频特征
        print("Extracting COVAREP acoustic features...")
        covarep_features = self.extract_covarep_acoustic_features(video_path)
        
        # 3. 提取OpenFace视觉特征
        print("Extracting OpenFace visual features...")
        openface_features = self.extract_openface_visual_features(video_path)
        
        # 4. 组织为MOSEI格式
        word_intervals = np.array([[i, i + 1] for i in range(len(word_features))], dtype=np.float32) if word_features else np.array([[0, 1]], dtype=np.float32)
        covarep_intervals = np.array([[i, i + 1] for i in range(len(covarep_features))], dtype=np.float32) if len(covarep_features) > 0 else np.array([[0, 1]], dtype=np.float32)
        openface_intervals = np.array([[i, i + 1] for i in range(len(openface_features))], dtype=np.float32) if len(openface_features) > 0 else np.array([[0, 1]], dtype=np.float32)

        mosei_data = {
            "language": {
                "features": word_features,  # [[word_embedding, start, end], ...]
                "intervals": word_intervals
            },
            "acoustic": {
                "features": covarep_features,  # [时间步数,40维]
                "intervals": covarep_intervals
            },
            "visual": {
                "features": openface_features,  # [时间步数, 709维] 
                "intervals": openface_intervals
            }
        }
        
        print(f"✓ Language: {len(word_features)} words with embeddings")
        print(f"✓ Acoustic: {covarep_features.shape} COVAREP features")
        print(f"✓ Visual: {openface_features.shape} OpenFace features")
        
        return mosei_data

def process_video_dataset_to_accurate_mosei_from_csv(csv_path, video_base_dir, output_dir, language="zh"):
    """
    根据您的Excel/CSV文件结构处理视频数据集，生成mmdatasdk兼容的.csd文件
    
    Args:
        csv_path (str): meta.csv或meta.xlsx文件路径
        video_base_dir (str): 视频文件根目录
        output_dir (str): 输出目录
        language (str): 语言代码
    """
    extractor = AccurateMOSEIExtractor(language=language)
    
    # 根据文件扩展名选择读取方式
    file_path = Path(csv_path)
    file_ext = file_path.suffix.lower()
    
    print(f"📊 读取数据文件: {csv_path}")
    
    try:
        if file_ext == '.csv':
            print("📋 检测到CSV文件，使用pandas.read_csv()...")
            # CSV也指定前两列为字符串
            df = pd.read_csv(csv_path, dtype={0: str, 1: str})
        else:
            print(f"⚠️ 未知文件格式 {file_ext}，尝试作为CSV读取...")
            df = pd.read_csv(csv_path, dtype={0: str, 1: str})
            
        print(f"✅ 数据文件读取成功: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 显示前几行的路径信息用于验证
        print("🔍 前3行路径信息验证:")
        for i in range(min(3, len(df))):
            video_id = str(df.iloc[i, 0]).strip()
            clip_id = str(df.iloc[i, 1]).strip()
            print(f"  行{i}: video_id='{video_id}', clip_id='{clip_id}' -> {video_id}/{clip_id}.mp4")
        
    except Exception as e:
        print(f"❌ 读取数据文件失败: {e}")
        print("💡 请确保:")
        print("  1. 文件路径正确")
        print("  2. 如果是Excel文件，请安装: pip install openpyxl")
        print("  3. 文件格式正确且可读")
        raise e
    
    video_base_dir = Path(video_base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建MOSEI兼容数据结构
    dataset = {
        "language": {},
        "acoustic": {},
        "visual": {},
        "labels": {}
    }
    
    # 处理每一行视频
    processed_count = 0
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing videos with {language} MOSEI features"):
        try:
            # 确保数据类型正确，并去除可能的空格
            video_id = str(row.iloc[0]).strip()
            clip_id = str(row.iloc[1]).strip()
            
            print(f"🔍 处理行{idx}: video_id='{video_id}', clip_id='{clip_id}'")
            
            # 处理标签数据，确保是数值类型
            try:
                overall_label = float(pd.to_numeric(row.iloc[2], errors='coerce'))
                text_label = float(pd.to_numeric(row.iloc[3], errors='coerce')) if len(row) > 3 else overall_label
                audio_label = float(pd.to_numeric(row.iloc[4], errors='coerce')) if len(row) > 4 else overall_label
                vis_label = float(pd.to_numeric(row.iloc[5], errors='coerce')) if len(row) > 5 else overall_label
                
                # 检查是否有无效值
                if pd.isna(overall_label):
                    print(f"⚠️ 行 {idx}: overall_label 无效，跳过")
                    continue
                    
            except (ValueError, TypeError) as e:
                print(f"⚠️ 行 {idx}: 标签数据转换失败 {e}，跳过")
                continue

            # 构建视频路径
            video_path = video_base_dir / video_id / f"{clip_id}.mp4"
            print(f"📁 构建视频路径: {video_path}")
            
            if not video_path.exists():
                print(f"⚠️ 视频文件不存在: {video_path}")
                
                # 详细检查路径问题
                parent_dir = video_path.parent
                if parent_dir.exists():
                    print(f"📂 父目录存在: {parent_dir}")
                    mp4_files = list(parent_dir.glob("*.mp4"))
                    print(f"📄 目录中的MP4文件: {[f.name for f in mp4_files]}")
                    
                    # 查看是否有相似的文件名
                    for mp4_file in mp4_files:
                        if mp4_file.stem == clip_id or mp4_file.stem == clip_id.lstrip('0'):
                            print(f"💡 可能的匹配文件: {mp4_file.name}")
                else:
                    print(f"📂 父目录不存在: {parent_dir}")
                
                error_count += 1
                continue

            print(f"🎬 处理视频: {video_id}/{clip_id}")
            mosei_data = extractor.process_video_to_accurate_mosei_format(str(video_path))
            segment_id = f"{video_id}_{clip_id}"

            dataset["language"][segment_id] = mosei_data["language"]
            dataset["acoustic"][segment_id] = mosei_data["acoustic"]  
            dataset["visual"][segment_id] = mosei_data["visual"]

            # 合并标签为一个多维标签
            num_labels = 4  # 标签的维度数量，例如整体情感、文本情感、音频情感、视觉情感
            dataset["labels"][segment_id] = {
                'features': np.array([[overall_label, text_label, audio_label, vis_label]], dtype=np.float32),
                'intervals': np.array([[i, i + 1] for i in range(num_labels)], dtype=np.int32)  # 动态生成 intervals
            }

            processed_count += 1
            print(f"✓ 成功处理: {video_id}/{clip_id} | 标签: O={overall_label:.2f}, T={text_label:.2f}, A={audio_label:.2f}, V={vis_label:.2f}")

        except Exception as e:
            error_count += 1
            print(f"✗ 处理错误 {video_id}/{clip_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n📊 处理统计:")
    print(f"  ✅ 成功处理: {processed_count} 个视频")
    print(f"  ❌ 处理失败: {error_count} 个视频")
    
    if processed_count == 0:
        print("❌ 没有成功处理任何视频，请检查:")
        print("  1. 视频文件路径是否正确")
        print("  2. 数据文件格式是否正确")
        print("  3. 标签数据是否有效")
        return dataset
    
    # 保存为mmdatasdk兼容的.csd格式文件
    print("💾 保存为mmdatasdk兼容的.csd格式...")
    metadata_acoustic = {
    "alignment compatible": True,
    "computational sequence description": "COVAREP Acoustic Features for CMU-MOSEI Dataset",
    "computational sequence version": "1.0",
    "contact": "abagherz@andrew.cmu.edu",
    "creator": "Amir Zadeh",
    "dataset bib citation": "@inproceedings{cmumoseiacl2018, title={Multimodal Language Analysis in the Wild: {CMU-MOSEI} Dataset and Interpretable Dynamic Fusion Graph}, author={Zadeh, Amir and Liang, Paul Pu and Vanbriesen, Jon and Poria, Soujanya and Cambria, Erik and Chen, Minghai and Morency, Louis-Philippe},booktitle={Association for Computational Linguistics (ACL)},year={2018}}",
    "dataset name": "CMU-MOSEI",
    "dataset version": "1.0",
    "dimension names": ['F0', 'VUV', 'NAQ', 'QOQ', 'H1H2', 'PSP', 'MDQ', 'peakSlope', 'Rd', 'Rd_conf', 'creak', 'MCEP_0', 'MCEP_1', 'MCEP_2', 'MCEP_3', 'MCEP_4', 'MCEP_5', 'MCEP_6', 'MCEP_7', 'MCEP_8', 'MCEP_9', 'MCEP_10', 'MCEP_11', 'MCEP_12', 'MCEP_13', 'MCEP_14', 'MCEP_15', 'MCEP_16', 'MCEP_17', 'MCEP_18', 'MCEP_19', 'MCEP_20', 'MCEP_21', 'MCEP_22', 'MCEP_23', 'MCEP_24', 'HMPDM_0', 'HMPDM_1', 'HMPDM_2', 'HMPDM_3', 'HMPDM_4', 'HMPDM_5', 'HMPDM_6', 'HMPDM_7', 'HMPDM_8', 'HMPDM_9', 'HMPDM_10', 'HMPDM_11', 'HMPDM_12', 'HMPDM_13', 'HMPDM_14', 'HMPDM_15', 'HMPDM_16', 'HMPDM_17', 'HMPDM_18', 'HMPDM_19', 'HMPDM_20', 'HMPDM_21', 'HMPDM_22', 'HMPDM_23', 'HMPDM_24', 'HMPDD_0', 'HMPDD_1', 'HMPDD_2', 'HMPDD_3', 'HMPDD_4', 'HMPDD_5', 'HMPDD_6', 'HMPDD_7', 'HMPDD_8', 'HMPDD_9', 'HMPDD_10', 'HMPDD_11', 'HMPDD_12'],
    "featureset bib citation": "@inproceedings{degottex2014covarep,title={COVAREP-A collaborative voice analysis repository for speech technologies},author={Degottex, Gilles and Kane, John and Drugman, Thomas and Raitio, Tuomo and Scherer, Stefan},booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2014 IEEE International Conference on},pages={960--964},year={2014},organization={IEEE}}",
    "md5": None,
    "root name": "COVAREP",
    "uuid": "af272a08-bb43-442d-b7d5-e7216a4c5119",
    }
    metadata_language = {
    "alignment compatible": True,
    "computational sequence description": "Word vector sequences for CMU-MOSEI Dataset",
    "computational sequence version": "1.0",
    "contact": "abagherz@andrew.cmu.edu",
    "creator": "Amir Zadeh",
    "dataset bib citation": "@inproceedings{cmumoseiacl2018, title={Multimodal Language Analysis in the Wild: {CMU-MOSEI} Dataset and Interpretable Dynamic Fusion Graph}, author={Zadeh, Amir and Liang, Paul Pu and Vanbriesen, Jon and Poria, Soujanya and Cambria, Erik and Chen, Minghai and Morency, Louis-Philippe},booktitle={Association for Computational Linguistics (ACL)},year={2018}}",
    "dataset name": "CMU-MOSEI",
    "dataset version": "1.0",
    "dimension names": ["vector"] * 300,  # 假设词向量维度为 300
    "featureset bib citation": "@article{P2FA,title={Speaker identification on the SCOTUS corpus},author={Yuan, Jiahong and Liberman, Mark},journal={Journal of the Acoustical Society of America},volume={123},number={5},pages={3878},year={2008},publisher={[New York: Acoustical Society of America]}}",
    "md5": None,
    "root name": "glove_vectors",
    "uuid": "8ac9704c-49b3-40ba-8c37-f029d3ddce43",
    }
    metadata_visual = {
    "alignment compatible": True,
    "computational sequence description": "FACET 4.2 Visual Features for CMU-MOSEI Dataset",
    "computational sequence version": "1.0",
    "contact": "abagherz@andrew.cmu.edu",
    "creator": "Amir Zadeh",
    "dataset bib citation": "@inproceedings{cmumoseiacl2018, title={Multimodal Language Analysis in the Wild: {CMU-MOSEI} Dataset and Interpretable Dynamic Fusion Graph}, author={Zadeh, Amir and Liang, Paul Pu and Vanbriesen, Jon and Poria, Soujanya and Cambria, Erik and Chen, Minghai and Morency, Louis-Philippe},booktitle={Association for Computational Linguistics (ACL)},year={2018}}",
    "dataset name": "CMU-MOSEI",
    "dataset version": "1.0",
    "dimension names": ['Anger', 'Contempt', 'Disgust', 'Joy', 'Fear', 'Baseline', 'Sadness', 'Surprise', 'Confusion', 'Frustration', 'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU18', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28', 'AU43', 'Has_Glasses', 'Is_Male', 'Pitch', 'Yaw', 'Roll'],
    "featureset bib citation": "@online{emotient,author = {iMotions},title = {Facial Expression Analysis},year = {2017},url = {goo.gl/1rh1JN}}",
    "md5": None,
    "root name": "FACET 4.2",
    "uuid": "f592e140-2766-426b-add3-8a14498059e7",
    }
    metadata_labels = {
    "alignment compatible": True,
    "computational sequence description": "Labels for CMU-MOSEI Dataset",
    "computational sequence version": "1.0",
    "contact": "abagherz@andrew.cmu.edu",
    "creator": "Amir Zadeh",
    "dataset bib citation": "@inproceedings{cmumoseiacl2018, title={Multimodal Language Analysis in the Wild: {CMU-MOSEI} Dataset and Interpretable Dynamic Fusion Graph}, author={Zadeh, Amir and Liang, Paul Pu and Vanbriesen, Jon and Poria, Soujanya and Cambria, Erik and Chen, Minghai and Morency, Louis-Philippe},booktitle={Association for Computational Linguistics (ACL)},year={2018}}",
    "dataset name": "CMU-MOSEI",
    "dataset version": "1.0",
    "dimension names": ['sentiment', 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear'],
    "featureset bib citation": "@online{amt, author = {Amazon},title = {Amazon Mechanical Turk},year = {2017},url = {https://www.mturk.com}}",
    "md5": None,
    "root name": "All Labels",
    "uuid": "bbce9ca9-e556-46f4-823e-7c5e0147afab",
    }

    extractor.save_to_csd_format(
        dataset["language"], 
        output_dir / "CMU_MOSEI_TimestampedWordVectors.csd",
        description="language",
        metadata=metadata_language
    )

    extractor.save_to_csd_format(
        dataset["acoustic"], 
        output_dir / "CMU_MOSEI_COVAREP.csd",
        description="acoustic",
        metadata=metadata_acoustic
    )

    extractor.save_to_csd_format(
        dataset["visual"], 
        output_dir / "CMU_MOSEI_VisualFacet42.csd",
        description="visual",
        metadata=metadata_visual
    )

    extractor.save_to_csd_format(
        dataset["labels"], 
        output_dir / "CMU_MOSEI_Labels.csd",
        description="labels",
        metadata=metadata_labels
    )
    
    language_model = "Chinese FastText/Word2Vec" if language in ["zh", "chinese", "中文"] else "English GloVe"
    
    print(f"\n✅ 处理完成!")
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 生成的文件:")
    print(f"  - CMU_MOSEI_TimestampedWordVectors.csd ({language_model})")
    print(f"  - CMU_MOSEI_COVAREP.csd") 
    print(f"  - CMU_MOSEI_VisualFacet42.csd")
    print(f"  - CMU_MOSEI_Labels.csd")
    
    # 测试兼容性
    print("\n🧪 测试mmdatasdk兼容性...")
    try:
        from mmsdk import mmdatasdk as md
        
        dataset_paths = {
            "language": str(output_dir / "CMU_MOSEI_TimestampedWordVectors.csd"),
            "acoustic": str(output_dir / "CMU_MOSEI_COVAREP.csd"),
            "visual": str(output_dir / "CMU_MOSEI_VisualFacet42.csd"),
            "labels": str(output_dir / "CMU_MOSEI_Labels.csd")
        }

        mosei_dataset = md.mmdataset(dataset_paths)
        
        print("✅ mmdatasdk加载成功!")
        
        # 显示数据集信息
        for modality in ["language", "acoustic", "visual", "labels"]:
            if modality in mosei_dataset:
                data = mosei_dataset[modality]
                print(f"  📊 {modality}: {len(data)} segments")
                
                if len(data) > 0:
                    first_key = list(data.keys())[0]
                    features = data[first_key]["features"]
                    intervals = data[first_key]["intervals"]
                    print(f"    样本 '{first_key}': features {features.shape}, intervals {intervals.shape}")
        
        print("🎉 可以在原MOSEI项目中使用!")
        
    except Exception as e:
        print(f"❌ mmdatasdk兼容性测试失败: {e}")
        print("请手动检查生成的.csd文件")
    
    return dataset

# 使用示例
if __name__ == "__main__":
    # 根据您的数据结构处理
    dataset = process_video_dataset_to_accurate_mosei_from_csv(
        csv_path="our_MSA/meta_test_only.csv",
        video_base_dir="our_MSA/ch_video",
        output_dir="our_MSA/ch_video_preprocess",
        language="zh"
    )