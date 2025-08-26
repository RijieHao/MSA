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
import shutil
import tempfile


# Ensure ffmpeg is available. If not found on PATH, try to use imageio_ffmpeg's bundled binary
def _ensure_ffmpeg_on_path():
    try:
        if shutil.which('ffmpeg'):
            return True
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg_exe and Path(ffmpeg_exe).exists():
                ffmpeg_path = Path(ffmpeg_exe)
                target_dir = Path(tempfile.gettempdir()) / "msaffmpeg"
                target_dir.mkdir(parents=True, exist_ok=True)
                target_ffmpeg = target_dir / "ffmpeg.exe"
                try:
                    if not target_ffmpeg.exists():
                        shutil.copy2(str(ffmpeg_path), str(target_ffmpeg))
                        try:
                            target_ffmpeg.chmod(target_ffmpeg.stat().st_mode | 0o111)
                        except Exception:
                            pass
                except Exception:
                    ffmpeg_dir = str(ffmpeg_path.parent)
                    os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
                    return True
                os.environ['PATH'] = str(target_dir) + os.pathsep + os.environ.get('PATH', '')
                return True
        except Exception:
            pass
        return False
    except Exception:
        return False


_FFMPEG_OK = _ensure_ffmpeg_on_path()
if not _FFMPEG_OK:
    print("Warning: ffmpeg not found on PATH. If you see FileNotFoundError when running, install ffmpeg or install the Python package 'imageio-ffmpeg'.")

class AccurateMOSEIExtractor:
    """Accurate MOSEI feature extractor.

    Implements feature extraction steps described in the MOSEI paper.
    Supports both Chinese and English processing.
    """
    
    def __init__(self, language="en"):
        self.language = language
        # Load Whisper for speech recognition and alignment
        self.whisper_model = whisper.load_model("base")

        # Choose word embedding model based on language
        if language in ["zh", "chinese", "中文"]:
            print("Initializing Chinese word embeddings...")
            self._load_chinese_embeddings()
        else:
            print("Initializing English GloVe embeddings...")
            self._load_english_embeddings()

        # Initialize MTCNN replacement (MediaPipe)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        # Initialize facial landmark detection (OpenFace alternative)
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
        """Load Chinese word embeddings (fallbacks to FastText or Word2Vec)."""
        try:
                # Option 1: try loading FastText Chinese model
                self._load_fasttext_chinese()
        except Exception as e:
            print(f"Failed to load FastText: {e}")
            try:
                # Option 2: try loading pretrained Chinese Word2Vec
                self._load_chinese_word2vec()
            except Exception as e2:
                print(f"Failed to load Word2Vec: {e2}")

    def _load_fasttext_chinese(self):
        """Load Chinese FastText model."""
        try:
            print("Loading FastText Chinese model...")
            # Download Chinese FastText model
            fasttext.util.download_model('zh', if_exists='ignore')
            self.word_model = fasttext.load_model('cc.zh.300.bin')
            self.word_embedding_dim = 300
            self.embedding_type = "fasttext"
            print("✓ Loaded FastText Chinese: 300d")
            
        except Exception as e:
            raise e

    def _load_chinese_word2vec(self):
        """Load a pretrained Chinese Word2Vec model from known local paths."""
        try:
            from gensim.models import KeyedVectors
            
            # Possible Chinese Word2Vec model paths
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
        """Load English GloVe embeddings using gensim.downloader."""
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
        """Extract word-level embedding features from a video.

        Returns a list of [word_embedding, start_time, end_time].
        """
    # 1. Extract audio from video
        video = VideoFileClip(str(video_path))
        temp_audio = f"temp_audio_for_{self.language}.wav"
        video.audio.write_audiofile(temp_audio, logger=None)

    # 2. Use Whisper for word-level alignment
        whisper_language = "zh" if self.language in ["zh", "chinese", "中文"] else self.language
        result = self.whisper_model.transcribe(
            temp_audio,
            language=whisper_language,
            word_timestamps=True
        )

    # 3. Convert to a sequence of word embeddings
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
                                # Chinese word segmentation
                                segmented_words = self._chinese_word_segmentation(word)
                                for seg_word in segmented_words:
                                    if seg_word.strip():
                                        word_embedding = self._get_chinese_word_embedding(seg_word)
                                        word_features.append([word_embedding, start, end])
                            else:
                                # English processing
                                word_clean = word.lower().strip()
                                word_embedding = self._get_english_word_embedding(word_clean)
                                word_features.append([word_embedding, start, end])

    # 4. Clean up temporary files
        try:
            os.remove(temp_audio)
        except Exception:
            pass
        video.close()

        return word_features

    def _chinese_word_segmentation(self, text):
        """Chinese word segmentation using jieba (fallback to character-level)."""
        try:
            words = list(jieba.cut(text, cut_all=False))
            return [w.strip() for w in words if w.strip()]
        except ImportError:
            print("Warning: jieba not available, using character-level segmentation")
            # If jieba is not available, fallback to character-level segmentation
            return list(text)

    def _get_chinese_word_embedding(self, word):
        """Return embedding vector for a Chinese word according to the selected embedding type."""
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
        """Get character-level average embedding for a word."""
        char_embeddings = []

        for char in word:
            if char in self.char_vocab:
                char_idx = self.char_vocab[char]
            else:
                char_idx = self.char_vocab.get('<UNK>', 0)

            char_embeddings.append(self.char_embeddings[char_idx])

        if char_embeddings:
            # Return the mean of character embeddings
            return np.mean(char_embeddings, axis=0).astype(np.float32)
        else:
            return np.zeros(self.word_embedding_dim, dtype=np.float32)

    def _get_english_word_embedding(self, word):
        """Return English word embedding (GloVe) or zero vector if not found."""
        if self.embedding_type == "glove" and self.word_model and word in self.word_model:
            return self.word_model[word].astype(np.float32)
        else:
            # If the token is not in GloVe, return zero vector
            return np.zeros(self.word_embedding_dim, dtype=np.float32)

    # Keep original audio and visual feature extraction methods
    def extract_covarep_acoustic_features(self, video_path):
        """Extract COVAREP-style acoustic features for the given video.

        Returns a numpy array of shape [time_steps, 40].
        """
        try:
            print(f"🔊 Extracting audio features: {video_path}")
            
            # 1. Extract audio from video
            video = VideoFileClip(str(video_path))
            frame_rate = video.fps  # get actual frame rate
            if video.audio is None:
                print(f"⚠️ Video has no audio track")
                video.close()
                return np.zeros((1, 40))
            
            temp_audio = "temp_audio_for_covarep.wav"
            
            # Force 22.05 kHz to avoid sampling rate issues
            video.audio.write_audiofile(
                temp_audio,
                fps=22050,  # Force 22.05 kHz
                logger=None,
            )
            video.close()
            
            # 2. Load audio using librosa, ensure sampling rate
            y, sr = librosa.load(temp_audio, sr=22050)  # Force 22.05 kHz
            
            print(f"🎵 Audio info: {len(y)/sr:.2f}s, sample rate: {sr}Hz")
            
            # 3. Compute number of frames (align with video)
            video_duration = len(y) / sr
            n_frames = max(1, int(video_duration * frame_rate))
            
            # 4. Extract features
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
            
            # 5. Clean up temporary files
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            acoustic_features = np.array(features_sequence)
            print(f"✅ Acoustic features shape: {acoustic_features.shape}")
            return acoustic_features
            
        except Exception as e:
            print(f"❌ Acoustic feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up temporary files
            temp_audio = "temp_audio_for_covarep.wav"
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return np.zeros((1, 40))
    
    def _extract_covarep_frame_features(self, y_segment, sr):
        try:
            # 1. 12 MFCC coefficients
            if len(y_segment) > 512:
                mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=12)
                mfcc_mean = np.mean(mfcc, axis=1) if mfcc.shape[1] > 0 else np.zeros(12)
                features.extend(mfcc_mean)
            else:
                features.extend([0.0] * 12)

            # 2. Pitch-related features - 8 dims
            try:
                if len(y_segment) > frame_length and frame_length > 0:
                    f0 = librosa.yin(
                        y_segment,
                        fmin=fmin,
                        fmax=min(400, sr // 4),
                        frame_length=frame_length,
                    )

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
                                len(f0_clean) / len(f0),
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

            # 3. Voiced/Unvoiced and spectral features - 6 dims
            if len(y_segment) > 512:
                try:
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y_segment))
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_segment, sr=sr))
                    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr))
                    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y_segment))
                    energy = np.sum(y_segment ** 2) / len(y_segment)

                    try:
                        harmonic, percussive = librosa.effects.hpss(y_segment)
                        hnr = np.sum(harmonic ** 2) / (np.sum(percussive ** 2) + 1e-8)
                    except Exception:
                        hnr = 1.0

                    voiced_unvoiced_features = [
                        zcr,
                        spectral_centroid,
                        spectral_bandwidth,
                        spectral_flatness,
                        energy,
                        hnr,
                    ]
                except Exception as vu_error:
                    print(f"⚠️ Voiced/Unvoiced features failed: {vu_error}")
                    voiced_unvoiced_features = [0.0] * 6
            else:
                voiced_unvoiced_features = [0.0] * 6

            features.extend(voiced_unvoiced_features)

            # 4. Peak slope parameters - 4 dims
            if len(y_segment) > 256:
                try:
                    hop_length = min(512, len(y_segment) // 4)
                    stft = librosa.stft(y_segment, hop_length=hop_length)
                    envelope = np.abs(stft)
                    envelope_mean = np.mean(envelope, axis=0)

                    if len(envelope_mean) > 2:
                        envelope_diff = np.diff(envelope_mean)
                        slope_features = [
                            np.mean(envelope_diff[envelope_diff > 0])
                            if np.any(envelope_diff > 0)
                            else 0.0,
                            np.mean(envelope_diff[envelope_diff < 0])
                            if np.any(envelope_diff < 0)
                            else 0.0,
                            np.std(envelope_diff),
                            np.max(np.abs(envelope_diff)),
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

            # 5. Maxima Dispersion Quotients - 4 dims
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
                            np.max(peak_intervals),
                        ]
                    else:
                        dispersion_features = [0.0] * 4
                except Exception as disp_error:
                    print(f"⚠️ Dispersion features failed: {disp_error}")
                    dispersion_features = [0.0] * 4
            else:
                dispersion_features = [0.0] * 4

            features.extend(dispersion_features)

            # 6. Additional emotion-related features to reach 40 dims
            remaining_dims = 40 - len(features)
            if remaining_dims > 0:
                try:
                    if len(y_segment) > 512:
                        hop_length = min(512, len(y_segment) // 4)
                        chroma = librosa.feature.chroma_stft(
                            y=y_segment, sr=sr, hop_length=hop_length
                        )
                        chroma_mean = (
                            np.mean(chroma, axis=1) if chroma.shape[1] > 0 else np.zeros(12)
                        )
                        additional_features = list(chroma_mean[:remaining_dims])
                    else:
                        additional_features = [0.0] * remaining_dims
                except Exception as chroma_error:
                    print(f"⚠️ Chroma features failed: {chroma_error}")
                    additional_features = [0.0] * remaining_dims

                features.extend(additional_features)

            # Ensure exactly 40 dims
            features = features[:40]
            while len(features) < 40:
                features.append(0.0)

        except Exception as e:
            print(f"❌ Error extracting COVAREP features: {e}")
            features = [0.0] * 40

        return np.array(features, dtype=np.float32)
    
    def extract_openface_visual_features(self, video_path):
        """
        Extract OpenFace-style visual features - reproduce original MOSEI visual features.
        Includes 68 facial landmarks, 20 shape parameters, HoG features, head pose, and eye gaze.

        Returns:
            np.ndarray: OpenFace-compatible features with shape [time_steps, 709]
        """
        cap = cv2.VideoCapture(video_path)
        features_sequence = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            # Replace MTCNN: use MediaPipe for face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_results = self.face_detection.process(rgb_frame)
            
            # Facial landmark detection
            mesh_results = self.face_mesh.process(rgb_frame)
            
            if mesh_results.multi_face_landmarks and detection_results.detections:
                landmarks = mesh_results.multi_face_landmarks[0]
                detection = detection_results.detections[0]
                frame_features = self._extract_openface_frame_features(
                    landmarks, detection, frame
                )
            else:
                # Use zero features when no face is detected
                frame_features = np.zeros(35)
            
            features_sequence.append(frame_features)
        
        cap.release()
        
        if len(features_sequence) == 0:
            features_sequence = [np.zeros(35)]
        
        return np.array(features_sequence)
    

    '''
    Below is a simplified version of the 709-d facial feature extraction.
    Uncommented lines refer to the original 709 dims; the final implementation uses 35 dims.
    '''
    def _extract_openface_frame_features(self, landmarks, detection, frame):
        """Extract single-frame OpenFace-style features (originally 709 dims, simplified here)."""
        features = []
        h, w = frame.shape[:2]
        
        # 1. 68 facial landmark coordinates (136 dims: 68 points × 2 coords)
        #landmark_coords = self._extract_68_facial_landmarks(landmarks)
        #features.extend(landmark_coords)  # 136 dims

        # 1. 10 key landmark coordinates (selected representative points) (20 dims)
        key_indices = [1, 33, 263, 61, 291, 199, 234, 454, 10, 152]  # nose tip, left/right eyes, mouth corners, cheeks, etc.
        for idx in key_indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y])
            else:
                features.extend([0.0, 0.0])

        # 2. 20 facial shape parameters (PCA-reduced shape descriptors)
        #shape_params = self._extract_facial_shape_parameters(landmarks)
        #features.extend(shape_params)  # 20 dims

        # 3. Facial HoG features (simplified to 3 dims)
        hog_features = self._extract_facial_hog_features(landmarks, frame)
        #features.extend(hog_features)  # 100 dims
        features.extend(hog_features[:3])  # 3 dims

        # 4. Head pose (6 dims: roll, pitch, yaw, x, y, z) (simplified to 3 dims)
        head_pose = self._extract_head_pose(landmarks, frame.shape)
        #features.extend(head_pose)  # 6 dims
        features.extend(head_pose[:3])  # 3 dims

        # 5. Eye gaze direction (4 dims)
        eye_gaze = self._extract_eye_gaze(landmarks)
        features.extend(eye_gaze)  # 4 dims

        # 6. FACS Action Units (17-dim AU intensities) (simplified to 5 dims)
        action_units = self._extract_facial_action_units(landmarks)
        #features.extend(action_units)  # 17 dims
        features.extend(action_units[:5])  # 5 dims

        # 7. Six basic emotions (Emotient FACET style)
        #basic_emotions = self._extract_basic_emotions(landmarks)
        #features.extend(basic_emotions)  # 6 dims

        # 8. Deep face embedding features (simplified)
        #face_embeddings = self._extract_face_embeddings(landmarks, frame)
        #features.extend(face_embeddings)  # 128 dims

        # 9. Additional features to pad to 709 dims
        #remaining_dims = 709 - len(features)
        #if remaining_dims > 0:
        #    additional_features = self._extract_additional_visual_features(
        #        landmarks, frame, remaining_dims
        #    )
        #    features.extend(additional_features)

        # Ensure exactly 709 (original) or 35 (simplified) dims
        #features = features[:709]
        features = features[:35]
        #while len(features) < 709:
        while len(features) < 35:
            features.append(0.0)

        return np.array(features, dtype=np.float32)
    
    def _extract_68_facial_landmarks(self, landmarks):
        """Extract 68 facial landmarks (OpenFace standard)."""
        # MediaPipe has 468 points; select the corresponding 68 OpenFace indices
        openface_68_indices = [
            # Chin contour (17 points: 0-16)
            172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
            # Right eyebrow (5 pts: 17-21)
            70, 63, 105, 66, 107,
            # Left eyebrow (5 pts: 22-26)
            55, 65, 52, 53, 46,
            # Nose (9 pts: 27-35)
            168, 8, 9, 10, 151, 195, 197, 196, 3,
            # Right eye (6 pts: 36-41)
            33, 7, 163, 144, 145, 153,
            # Left eye (6 pts: 42-47)
            362, 382, 381, 380, 374, 373,
            # Outer mouth contour (12 pts: 48-59)
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            # Inner mouth contour (8 pts: 60-67)
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

        return coords[:136]
    
    def _extract_facial_shape_parameters(self, landmarks):
        """Extract 20 facial shape parameters (PCA-style reduction)."""
        # Simplified version: compute shape descriptors from keypoints
        params = []
        # Face width-height ratio
        face_width = abs(landmarks.landmark[234].x - landmarks.landmark[454].x)
        face_height = abs(landmarks.landmark[10].y - landmarks.landmark[152].y)
        params.append(face_width / (face_height + 1e-8))

        # Eye width ratios
        left_eye_width = abs(landmarks.landmark[33].x - landmarks.landmark[133].x)
        right_eye_width = abs(landmarks.landmark[362].x - landmarks.landmark[263].x)
        params.extend([left_eye_width, right_eye_width])

        # Mouth parameters
        mouth_width = abs(landmarks.landmark[61].x - landmarks.landmark[291].x)
        mouth_height = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
        params.extend([mouth_width, mouth_height])

        # Nose parameters
        nose_width = abs(landmarks.landmark[125].x - landmarks.landmark[141].x)
        nose_height = abs(landmarks.landmark[19].y - landmarks.landmark[1].y)
        params.extend([nose_width, nose_height])
        
        # Pad to 20 dims
        while len(params) < 20:
            params.append(0.0)

        return params[:20]
    
    def _extract_facial_hog_features(self, landmarks, frame):
        """Extract facial HoG features (simplified)."""
        try:
            
            # Extract face region
            face_region = self._extract_face_region(landmarks, frame)
            
            if face_region is not None:
                # Convert to grayscale
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                # Resize face region
                gray_face = cv2.resize(gray_face, (64, 64))
                
                # Extract HoG features
                hog_features = hog(
                    gray_face,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys',
                    feature_vector=True
                )
                
                # Reduce/Pad to 100 dims
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
        """Extract face region from the frame using landmark bounding box."""
        try:
            h, w = frame.shape[:2]
            
            # Compute face bounding box
            x_coords = [landmarks.landmark[i].x * w for i in range(len(landmarks.landmark))]
            y_coords = [landmarks.landmark[i].y * h for i in range(len(landmarks.landmark))]
            
            x1, x2 = int(min(x_coords)), int(max(x_coords))
            y1, y2 = int(min(y_coords)), int(max(y_coords))
            
            # Add margin
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
        """Extract head pose (6 dims)."""
        # Simplified head pose estimation from landmarks
        try:
            nose_tip = landmarks.landmark[1]
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[362]

            # Roll (head tilt)
            eye_angle = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
            roll = np.degrees(eye_angle)

            # Pitch (nod up/down)
            pitch = (nose_tip.y - 0.5) * 60

            # Yaw (turn left/right)
            face_center_x = (left_eye.x + right_eye.x) / 2
            yaw = (nose_tip.x - face_center_x) * 120

            # Position (relative to image center)
            h, w = frame_shape[:2]
            x_pos = (nose_tip.x - 0.5) * w
            y_pos = (nose_tip.y - 0.5) * h
            z_pos = abs(landmarks.landmark[10].y - landmarks.landmark[152].y) * 100

            return [roll, pitch, yaw, x_pos, y_pos, z_pos]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _extract_eye_gaze(self, landmarks):
        """Extract eye gaze direction (4 dims)."""
        # Simplified gaze estimation
        try:
            left_eye_center = np.mean(
                [[landmarks.landmark[i].x, landmarks.landmark[i].y] for i in [33, 133]], axis=0
            )
            right_eye_center = np.mean(
                [[landmarks.landmark[i].x, landmarks.landmark[i].y] for i in [362, 263]], axis=0
            )

            # compute gaze direction
            left_gaze_x = (left_eye_center[0] - 0.3) * 2
            left_gaze_y = (left_eye_center[1] - 0.4) * 2
            right_gaze_x = (right_eye_center[0] - 0.7) * 2
            right_gaze_y = (right_eye_center[1] - 0.4) * 2

            return [left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]
    
    def _extract_facial_action_units(self, landmarks):
        """Estimate FACS action unit intensities (17 dims) from landmarks."""
        # Estimate main AUs based on facial keypoints
        aus = []

        # AU1 - Inner brow raise
        au1 = max(0, 0.5 - landmarks.landmark[55].y) * 10
        aus.append(au1)

        # AU2 - Outer brow raise
        au2 = max(0, 0.4 - landmarks.landmark[70].y) * 10
        aus.append(au2)

        # AU4 - Brow furrow
        brow_distance = abs(landmarks.landmark[55].x - landmarks.landmark[70].x)
        au4 = max(0, 0.1 - brow_distance) * 50
        aus.append(au4)

        # AU5 - Upper eyelid raise
        left_eye_open = abs(landmarks.landmark[33].y - landmarks.landmark[145].y)
        au5 = left_eye_open * 20
        aus.append(au5)

        # AU6 - Cheek raise
        cheek_height = landmarks.landmark[116].y
        au6 = max(0, 0.6 - cheek_height) * 15
        aus.append(au6)

        # AU9 - Nose wrinkle
        nose_width = abs(landmarks.landmark[125].x - landmarks.landmark[141].x)
        au9 = max(0, nose_width - 0.02) * 100
        aus.append(au9)

        # AU12 - Lip corner puller (smile)
        mouth_corner_avg = (landmarks.landmark[61].y + landmarks.landmark[84].y) / 2
        mouth_center = landmarks.landmark[13].y
        au12 = max(0, mouth_center - mouth_corner_avg) * 50
        aus.append(au12)

        # AU15 - Lip corner depressor
        au15 = max(0, mouth_corner_avg - mouth_center) * 50
        aus.append(au15)

        # AU20 - Lip stretcher (horizontal)
        mouth_width = abs(landmarks.landmark[61].x - landmarks.landmark[291].x)
        au20 = mouth_width * 100
        aus.append(au20)

        # AU25 - Lips part
        mouth_open = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
        au25 = mouth_open * 100
        aus.append(au25)

        # Pad to 17 dims
        while len(aus) < 17:
            aus.append(0.0)

        return aus[:17]
    
    def _extract_basic_emotions(self, landmarks):
        """Extract six basic emotions (Emotient FACET style) using landmarks."""
        # Simplified emotion recognition from facial keypoints
        emotions = []

        # Happiness - based on mouth corner lift
        mouth_corner_lift = self._calculate_mouth_corner_lift(landmarks)
        happiness = max(0, mouth_corner_lift) * 5
        emotions.append(happiness)

        # Sadness - based on mouth corner drop and brow drop
        mouth_corner_drop = -min(0, mouth_corner_lift)
        brow_drop = max(0, 0.45 - landmarks.landmark[55].y)
        sadness = (mouth_corner_drop + brow_drop) * 3
        emotions.append(sadness)

        # Anger - based on brow furrow
        brow_furrow = max(0, 0.1 - abs(landmarks.landmark[55].x - landmarks.landmark[70].x))
        anger = brow_furrow * 20
        emotions.append(anger)

        # Disgust - based on nose scrunch
        nose_scrunch = abs(landmarks.landmark[125].x - landmarks.landmark[141].x)
        disgust = max(0, nose_scrunch - 0.02) * 50
        emotions.append(disgust)

        # Surprise - based on brow raise and mouth opening
        brow_raise = max(0, 0.4 - landmarks.landmark[70].y)
        mouth_open = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
        surprise = (brow_raise + mouth_open) * 10
        emotions.append(surprise)

        # Fear - based on wide eyes and brow raise
        eye_wide = abs(landmarks.landmark[33].y - landmarks.landmark[145].y)
        fear = (eye_wide + brow_raise) * 8
        emotions.append(fear)

        return emotions
    
    def _calculate_mouth_corner_lift(self, landmarks):
        """Compute mouth corner lift measure."""
        left_corner = landmarks.landmark[61].y
        right_corner = landmarks.landmark[84].y
        mouth_center = landmarks.landmark[13].y

        corner_avg = (left_corner + right_corner) / 2
        return mouth_center - corner_avg
    
    def _extract_face_embeddings(self, landmarks, frame):
        """Extract deep face embedding (simplified placeholder)."""
        # A proper face recognition model (e.g., FaceNet) can be integrated here.
        # Currently we use a simplified keypoint-based embedding.
        embedding = []
        
    # Compute geometric features from keypoints as embedding
        for i in range(0, min(128, len(landmarks.landmark))):
            if i < len(landmarks.landmark):
                landmark = landmarks.landmark[i]
                embedding.extend([landmark.x, landmark.y])
            else:
                embedding.extend([0.0, 0.0])
        
    # Pad/trim to 128 dims
        while len(embedding) < 128:
            embedding.append(0.0)
        
        return embedding[:128]
    
    def _extract_additional_visual_features(self, landmarks, frame, num_features):
        """Extract additional visual features to reach 709 dims (fallback/simple features)."""
        features = []

        # Add more geometric features as simple combinations of landmark coords
        for i in range(num_features):
            if i < len(landmarks.landmark):
                landmark = landmarks.landmark[i % len(landmarks.landmark)]
                features.append(landmark.x * landmark.y)
            else:
                features.append(0.0)

        return features[:num_features]
    
    def save_to_csd_format(self, data, output_path, description="", metadata={}):
        with h5py.File(output_path, 'w') as f:
            # create top-level group 'computational_sequences'
            top_group = f.create_group("computational_sequences")

            # create 'data' group
            data_group = top_group.create_group("data")
            for segment_id, segment_data in data.items():
                features = np.array(segment_data.get("features", []), dtype=np.float32)
                intervals = np.array(segment_data.get("intervals", []), dtype=np.float32)
                seg_group = data_group.create_group(segment_id)
                seg_group.create_dataset("features", data=features, compression='gzip')
                seg_group.create_dataset("intervals", data=intervals, compression='gzip')

            # create 'metadata' group
            metadata_group = top_group.create_group("metadata")
            for key, value in metadata.items():
                try:
                    if key == "md5" and value is None:
                        value = ""
                    elif value is None or isinstance(value, (list, tuple, dict)):
                        value = str(value)
                    metadata_group.attrs[key] = value
                except Exception as e:
                    print(f"❌ Error setting metadata attribute {key}: {e}")

            # Add the description attribute
            metadata_group.attrs["description"] = description

        print(f"✅ CSD file saved successfully: {output_path}")

    def process_labels_csv_to_csd(self, csv_path):
        """
        Convert your meta.csv into MOSEI-style label data.

        Args:
            csv_path (str): Path to meta.csv

        Returns:
            dict: Dictionary containing labels data
        """
        print(f"Processing labels from {csv_path}")

        # Read CSV file
        df = pd.read_csv(csv_path)

        labels_data = {}

        for _, row in df.iterrows():
            try:
                # Based on your CSV format:
                # Column 1: video_id (folder name)
                # Column 2: clip_id (file name)
                # Column 3: overall label
                # Column 4: text label (optional)
                # Column 5: audio label (optional)
                # Column 6: visual label (optional)

                video_id = str(row.iloc[0])     # video_0001
                clip_id = str(row.iloc[1])      # 0001
                overall_label = float(row.iloc[2])  # label

                # Use video_id + clip_id as the segment_id to match the video file
                segment_id = f"{video_id}_{clip_id}"

                # Create MOSEI-style label structure using overall label
                labels_data[segment_id] = {
                    'features': np.array([[overall_label]], dtype=np.float32),
                    'intervals': np.array([[0, 1]], dtype=np.int32)
                }

            except Exception as e:
                print(f"Error processing label for row {_}: {e}")
                # Use a default neutral label on error
                segment_id = f"error_{_}"
                labels_data[segment_id] = {
                    'features': np.array([[0.0]], dtype=np.float32),
                    'intervals': np.array([[0, 1]], dtype=np.int32)
                }

        print(f"✓ Converted {len(labels_data)} labels")
        return labels_data

    def process_video_to_accurate_mosei_format(self, video_path):
        """
        Convert an MP4 video into a more accurate MOSEI dataset format.

        Returns:
            dict: Dictionary containing accurate features for the three modalities
        """
        print(f"Processing video with accurate MOSEI features: {video_path}")

        # 1. Extract word embedding sequence
        print("Extracting word-based language features...")
        word_features = self.extract_word_features(video_path)

        # 2. Extract COVAREP acoustic features
        print("Extracting COVAREP acoustic features...")
        covarep_features = self.extract_covarep_acoustic_features(video_path)

        # 3. Extract OpenFace visual features
        print("Extracting OpenFace visual features...")
        openface_features = self.extract_openface_visual_features(video_path)

        # 4. Organize into MOSEI format
        word_intervals = (
            np.array([[i, i + 1] for i in range(len(word_features))], dtype=np.float32)
            if word_features
            else np.array([[0, 1]], dtype=np.float32)
        )
        covarep_intervals = (
            np.array([[i, i + 1] for i in range(len(covarep_features))], dtype=np.float32)
            if len(covarep_features) > 0
            else np.array([[0, 1]], dtype=np.float32)
        )
        openface_intervals = (
            np.array([[i, i + 1] for i in range(len(openface_features))], dtype=np.float32)
            if len(openface_features) > 0
            else np.array([[0, 1]], dtype=np.float32)
        )

        mosei_data = {
            "language": {"features": word_features, "intervals": word_intervals},
            "acoustic": {"features": covarep_features, "intervals": covarep_intervals},
            "visual": {"features": openface_features, "intervals": openface_intervals},
        }

        print(f"✓ Language: {len(word_features)} words with embeddings")
        try:
            print(f"✓ Acoustic: {covarep_features.shape} COVAREP features")
        except Exception:
            print("✓ Acoustic: unavailable")
        try:
            print(f"✓ Visual: {openface_features.shape} OpenFace features")
        except Exception:
            print("✓ Visual: unavailable")

        return mosei_data

def process_video_dataset_to_accurate_mosei_from_csv(csv_path, video_base_dir, output_dir, language="zh"):
    """
    Process a video dataset described by an Excel/CSV file and generate mmdatasdk-compatible .csd files.

    Args:
        csv_path (str): Path to meta.csv or meta.xlsx
        video_base_dir (str): Root directory containing video files
        output_dir (str): Output directory for generated .csd files
        language (str): Language code (e.g., 'zh' or 'en')
    """
    extractor = AccurateMOSEIExtractor(language=language)
    
    # 根据文件扩展名选择读取方式
    file_path = Path(csv_path)
    file_ext = file_path.suffix.lower()
    
    print(f"📊 Reading data file: {csv_path}")
    
    try:
        if file_ext == '.csv':
            print("📋 CSV detected, using pandas.read_csv()...")
            # For CSV, ensure the first two columns are read as strings
            df = pd.read_csv(csv_path, dtype={0: str, 1: str})
            print(f"✅ Data file read OK: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"⚠️ Unknown file extension {file_ext}, attempting to read as CSV...")
            df = pd.read_csv(csv_path, dtype={0: str, 1: str})
            

        # Show first few rows to verify paths
        print("🔍 First 3 rows path verification:")
        for i in range(min(3, len(df))):
            video_id = str(df.iloc[i, 0]).strip()
            clip_id = str(df.iloc[i, 1]).strip()
            print(f"  row{i}: video_id='{video_id}', clip_id='{clip_id}' -> {video_id}/{clip_id}.mp4")
        
    except Exception as e:
        print(f"❌ Failed to read data file: {e}")
        print("💡 Please ensure:")
        print("  1. The file path is correct")
        print("  2. If Excel, install: pip install openpyxl")
        print("  3. The file format is valid and readable")
        raise e
    
    video_base_dir = Path(video_base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create MOSEI-compatible data structure
    dataset = {
        "language": {},
        "acoustic": {},
        "visual": {},
        "labels": {}
    }
    
    # Process each video row
    processed_count = 0
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing videos with {language} MOSEI features"):
        try:
            # Ensure correct data types and strip possible whitespace
            video_id = str(row.iloc[0]).strip()
            clip_id = str(row.iloc[1]).strip()
            
            print(f"🔍 Processing row {idx}: video_id='{video_id}', clip_id='{clip_id}'")
            
            # Process label data, ensure numeric types
            try:
                overall_label = float(pd.to_numeric(row.iloc[2], errors='coerce'))
                text_label = float(pd.to_numeric(row.iloc[3], errors='coerce')) if len(row) > 3 else overall_label
                audio_label = float(pd.to_numeric(row.iloc[4], errors='coerce')) if len(row) > 4 else overall_label
                vis_label = float(pd.to_numeric(row.iloc[5], errors='coerce')) if len(row) > 5 else overall_label
                
                # Check for invalid values
                if pd.isna(overall_label):
                    print(f"⚠️ Row {idx}: overall_label invalid, skipping")
                    continue
                    
            except (ValueError, TypeError) as e:
                print(f"⚠️ Row {idx}: label conversion failed {e}, skipping")
                continue

            # Construct video path
            video_path = video_base_dir / video_id / f"{clip_id}.mp4"
            print(f"📁 Constructed video path: {video_path}")
            
            if not video_path.exists():
                print(f"⚠️ Video file not found: {video_path}")
                
                # Inspect path details
                parent_dir = video_path.parent
                if parent_dir.exists():
                    print(f"📂 Parent directory exists: {parent_dir}")
                    mp4_files = list(parent_dir.glob("*.mp4"))
                    print(f"📄 MP4 files in directory: {[f.name for f in mp4_files]}")
                    
                    # Look for similar file names
                    for mp4_file in mp4_files:
                        if mp4_file.stem == clip_id or mp4_file.stem == clip_id.lstrip('0'):
                            print(f"💡 Possible matching file: {mp4_file.name}")
                else:
                    print(f"📂 Parent directory missing: {parent_dir}")
                
                error_count += 1
                continue

            print(f"🎬 Processing video: {video_id}/{clip_id}")
            mosei_data = extractor.process_video_to_accurate_mosei_format(str(video_path))
            segment_id = f"{video_id}_{clip_id}"

            dataset["language"][segment_id] = mosei_data["language"]
            dataset["acoustic"][segment_id] = mosei_data["acoustic"]  
            dataset["visual"][segment_id] = mosei_data["visual"]

            # Combine labels into a multi-dimensional label vector
            num_labels = 4  # Number of label dimensions: overall, text, audio, visual
            dataset["labels"][segment_id] = {
                'features': np.array([[overall_label, text_label, audio_label, vis_label]], dtype=np.float32),
                'intervals': np.array([[i, i + 1] for i in range(num_labels)], dtype=np.int32)  # dynamically generate intervals
            }

            processed_count += 1
            print(f"✓ Successfully processed: {video_id}/{clip_id} | labels: O={overall_label:.2f}, T={text_label:.2f}, A={audio_label:.2f}, V={vis_label:.2f}")

        except Exception as e:
            error_count += 1
            print(f"✗ Processing error {video_id}/{clip_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n📊 Processing summary:")
    print(f"  ✅ Successfully processed: {processed_count} videos")
    print(f"  ❌ Failed: {error_count} videos")
    
    if processed_count == 0:
        print("❌ No videos were successfully processed. Please check:")
        print("  1. Are the video file paths correct?")
        print("  2. Is the data file format valid?")
        print("  3. Are label values valid?")
        return dataset
    
    # Save as mmdatasdk-compatible .csd files
    print("💾 Saving as mmdatasdk-compatible .csd files...")
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
    "dimension names": ["vector"] * 300,  # assume word vector dimension is 300
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

    print(f"\n✅ Processing complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Generated files:")
    print(f"  - CMU_MOSEI_TimestampedWordVectors.csd ({language_model})")
    print(f"  - CMU_MOSEI_COVAREP.csd") 
    print(f"  - CMU_MOSEI_VisualFacet42.csd")
    print(f"  - CMU_MOSEI_Labels.csd")
    
    # Test compatibility with mmdatasdk
    print("\n🧪 Testing mmdatasdk compatibility...")
    try:
        from mmsdk import mmdatasdk as md

        dataset_paths = {
            "language": str(output_dir / "CMU_MOSEI_TimestampedWordVectors.csd"),
            "acoustic": str(output_dir / "CMU_MOSEI_COVAREP.csd"),
            "visual": str(output_dir / "CMU_MOSEI_VisualFacet42.csd"),
            "labels": str(output_dir / "CMU_MOSEI_Labels.csd"),
        }

        mosei_dataset = md.mmdataset(dataset_paths)

        print("✅ mmdatasdk loaded successfully!")

        # Show dataset info
        for modality in ["language", "acoustic", "visual", "labels"]:
            if modality in mosei_dataset:
                data = mosei_dataset[modality]
                print(f"  📊 {modality}: {len(data)} segments")

                if len(data) > 0:
                    first_key = list(data.keys())[0]
                    features = data[first_key]["features"]
                    intervals = data[first_key]["intervals"]
                    print(f"    Sample '{first_key}': features {features.shape}, intervals {intervals.shape}")

        print("🎉 Compatible with the original MOSEI project!")

    except Exception as e:
        print(f"❌ mmdatasdk compatibility test failed: {e}")
        print("Please manually inspect the generated .csd files")

    return dataset


if __name__ == "__main__":

    dataset = process_video_dataset_to_accurate_mosei_from_csv(
        csv_path="our_MSA/meta_test_only.csv",
        video_base_dir="our_MSA/ch_video",
        output_dir="our_MSA/ch_video_preprocess",
        language="zh"
    )