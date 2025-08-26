import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from moviepy.video.io.VideoFileClip import VideoFileClip
#from moviepy import VideoFileClip
import whisper
import librosa
import cv2
import mediapipe as mp
import torch
import shutil
import tempfile
from scipy.signal import find_peaks
from skimage.feature import hog
import csv  # add csv module


# Ensure ffmpeg is available. If not found on PATH, try to use imageio_ffmpeg's bundled binary
def _ensure_ffmpeg_on_path():
    try:
        # Quick check: is ffmpeg already callable?
        if shutil.which('ffmpeg'):
            return True
        # Try imageio_ffmpeg package which may provide a bundled ffmpeg binary
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg_exe and Path(ffmpeg_exe).exists():
                ffmpeg_path = Path(ffmpeg_exe)
                # If the binary file is not named 'ffmpeg.exe', copy it to a temp dir as ffmpeg.exe
                target_dir = Path(tempfile.gettempdir()) / "msaffmpeg"
                target_dir.mkdir(parents=True, exist_ok=True)
                target_ffmpeg = target_dir / "ffmpeg.exe"
                try:
                    if not target_ffmpeg.exists():
                        # Copy the bundled binary to target location as ffmpeg.exe
                        shutil.copy2(str(ffmpeg_path), str(target_ffmpeg))
                        try:
                            # Make executable (no-op on Windows but safe)
                            target_ffmpeg.chmod(target_ffmpeg.stat().st_mode | 0o111)
                        except Exception:
                            pass
                except Exception:
                    # fallback: just add original parent dir to PATH
                    ffmpeg_dir = str(ffmpeg_path.parent)
                    os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
                    return True

                # Prepend our temp dir to PATH so subprocess calls can find 'ffmpeg'
                os.environ['PATH'] = str(target_dir) + os.pathsep + os.environ.get('PATH', '')
                return True
        except Exception:
            # imageio_ffmpeg not available or failed
            pass
        return False
    except Exception:
        return False


# Run check on import and provide a helpful warning if ffmpeg is not available
_FFMPEG_OK = _ensure_ffmpeg_on_path()
if not _FFMPEG_OK:
    # Lightweight warning printed on import — keep small to avoid noisy logs
    print("Warning: ffmpeg not found on PATH. Whisper/MoviePy may fail to process audio/video.\n"
          "If you see FileNotFoundError when running, install ffmpeg or install the Python package 'imageio-ffmpeg'.")


# Configuration parameters
TEXT_EMBEDDING_DIM = 768
AUDIO_FEATURE_SIZE = 40
VISUAL_FEATURE_SIZE = 35
SEED = 42

class MOSEIExtractor:
    def __init__(self, language="unknown"):
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize Whisper model
        self.whisper_model = whisper.load_model("base").to(self.device)

    # initialize BERT models
     # ===== load English BERT =====
        self.en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.en_bert = BertModel.from_pretrained("bert-base-uncased").to(self.device).eval()

    # ===== load Chinese BERT =====
        self.zh_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.zh_bert = BertModel.from_pretrained("bert-base-chinese").to(self.device).eval()

    # initialize MediaPipe face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )

    def find_audio_file(self, video_path, audio_dir):
        """Find corresponding audio file in `audio_dir` by video filename.

        Returns the Path to a .wav file with the same stem as the video, or None if not found.
        """
        video_name = Path(video_path).stem  # video filename without extension
        audio_path = Path(audio_dir) / f"{video_name}.wav"  # assume .wav format
        if audio_path.exists():
            return audio_path
        else:
            return None

    # Detect language
    def detect_language(self, audio_path):
        """Detect audio language using the Whisper model and return language code."""
        result = self.whisper_model.transcribe(audio_path, task="lang")
        detected_language = result["language"]
        return detected_language


# ------------------------------------- Text feature extraction -------------------------------------
    def extract_text_features(self, video_path, audio_path=None):
        """Extract text features from a video or audio file and produce BERT embeddings."""
        # If an audio path is provided, use it directly
        video = VideoFileClip(str(video_path))
        if audio_path:
            temp_audio = audio_path
        else:
            # extract audio from video
            temp_audio = "temp_audio.wav"
            video.audio.write_audiofile(temp_audio, logger=None)

        # If language is unknown, detect it first
        if self.language == "unknown":
            self.language = self.detect_language(temp_audio)

        # use Whisper to transcribe
        result = self.whisper_model.transcribe(str(temp_audio), language=self.language, word_timestamps=True)
        if not audio_path:  # remove temporary audio file if created
            os.remove(temp_audio)

        # concatenate words into a single string
        words = [word["word"] for segment in result["segments"] for word in segment["words"]]
        text_str = " ".join(words)

        if self.language.startswith("zh"):  # Chinese
            tokenizer = self.zh_tokenizer
            bert_model = self.zh_bert
        else:  # default English
            tokenizer = self.en_tokenizer
            bert_model = self.en_bert
        # extract BERT embeddings
        inputs = tokenizer(text_str, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return self.language, text_str, embedding[0]  # return 1D vector

#-------------------------------------Audio feature extraction-------------------------------------
    def extract_audio_features(self, video_path, audio_path=None):
        """Extract audio features from a video or audio file and produce high-level features"""
        # If an audio path is provided, use it directly
        video = VideoFileClip(str(video_path))
        if audio_path:
            temp_audio = audio_path
        else:
            # extract audio from video
            temp_audio = "temp_audio.wav"
            video.audio.write_audiofile(temp_audio, logger=None)

        y, sr = librosa.load(temp_audio, sr=22050)
        if not audio_path:  # if a temporary audio file, remove it
            os.remove(temp_audio)

        # extract frame-level audio features
        hop_length = max(1, len(y) // int(video.duration * video.fps))
        features = []
        for i in range(0, len(y), hop_length):
            frame = y[i:i + hop_length]
            if len(frame) > 256:
                frame_features = self._extract_covarep_frame_features(frame, sr)
                features.append(frame_features)
            else:
                features.append(np.zeros(AUDIO_FEATURE_SIZE))
        features = np.array(features)

    # Generate high-level features
        return np.mean(features, axis=0) if features.size > 0 else np.zeros(AUDIO_FEATURE_SIZE)
    def _extract_covarep_frame_features(self, y_segment, sr):
        """Extract features for a single audio frame"""
        features = []
        fmin = 85
        frame_length = max(369, min(369, len(y_segment) // 2))
        # 12 MFCC coefficients
        if len(y_segment) > 512:
            mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=12)
            mfcc = np.nan_to_num(mfcc, nan=0.0)
            features.extend(np.mean(mfcc, axis=1) if mfcc.shape[1] > 0 else np.zeros(12))
        else:
            features.extend([0.0] * 12)

    # pitch-related features (8 dims)
        if len(y_segment) > frame_length and frame_length > 0:
            f0 = librosa.yin(y_segment, fmin=85, fmax=min(400, sr//4), frame_length=frame_length)
            f0 = np.nan_to_num(f0, nan=0.0)
            f0_clean = f0[f0 > 0]
            if len(f0_clean) > 0:
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
            else:
                pitch_features = [0.0] * 8
        else:
            pitch_features = [0.0] * 8
        features.extend(pitch_features)

    # voiced/unvoiced and spectral features (6 dims)
        if len(y_segment) > 512:
            # Zero-crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(y_segment))
            # Spectral centroid
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_segment, sr=sr))
            # Spectral bandwidth
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr))
            # Spectral flatness - fixed API call
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y_segment))
            # Energy
            energy = np.sum(y_segment ** 2) / len(y_segment)
            harmonic, percussive = librosa.effects.hpss(y_segment)
            hnr = np.sum(harmonic ** 2) / (np.sum(percussive ** 2) + 1e-8)
            features.extend([
                zcr, spectral_centroid, spectral_bandwidth, 
                spectral_flatness, energy, hnr
            ])
        else:
            features.extend([0.0] * 6)


    # 5. Peak slope parameters - 4 dims
        if len(y_segment) > 256:
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
        else:
            slope_features = [0.0] * 4     
        features.extend(slope_features)

    # 6. Maxima Dispersion Quotients - 4 dims
        if len(y_segment) > 512:
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
        else:
            dispersion_features = [0.0] * 4
        features.extend(dispersion_features)

    # fill remaining emotional features to reach 40 dims
        remaining_dims = AUDIO_FEATURE_SIZE - len(features)
        if len(y_segment) > 512:
            chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr, hop_length=min(512, len(y_segment)//4))
            chroma_mean = np.mean(chroma, axis=1) if chroma.shape[1] > 0 else np.zeros(12)
            additional_features = list(chroma_mean[:remaining_dims])
        else:
            additional_features = [0.0] * remaining_dims
        features.extend(additional_features)
        
        features = np.nan_to_num(features, nan=0.0)
        features = features[:AUDIO_FEATURE_SIZE]
        while len(features) < AUDIO_FEATURE_SIZE:
            features.append(0.0)
        return np.array(features)
    

#-------------------------------------visual feature extraction------------------------------------------------------
    def extract_visual_features(self, video_path):
        """Extract visual features from a video and produce high-level features"""
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

    # Generate high-level features
        return np.mean(features, axis=0) if features.size > 0 else np.zeros(VISUAL_FEATURE_SIZE)

    def _extract_visual_frame_features(self, landmarks, frame):
        """Extract features for a single video frame"""
        features = []
        # 10 key facial landmark points -> 20 dims
        key_indices = [1, 33, 263, 61, 291, 199, 234, 454, 10, 152]
        for idx in key_indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y])
            else:
                features.extend([0.0, 0.0])

    # HoG features (3 dims)
        h, w = frame.shape[:2]
    # get face bounding box
        x_coords = [landmarks.landmark[i].x * w for i in range(len(landmarks.landmark))]
        y_coords = [landmarks.landmark[i].y * h for i in range(len(landmarks.landmark))]
        x1, x2 = int(min(x_coords)), int(max(x_coords))
        y1, y2 = int(min(y_coords)), int(max(y_coords))
    # add margin
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

    # head pose (simplified to 3 dims)
        nose_tip = landmarks.landmark[1]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[362]
    # Roll (head tilt)
        eye_angle = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
        roll = np.degrees(eye_angle)
    # Pitch (pitch)
        pitch = (nose_tip.y - 0.5) * 60
    # Yaw (yaw)
        face_center_x = (left_eye.x + right_eye.x) / 2
        yaw = (nose_tip.x - face_center_x) * 120
        head_pose = [roll, pitch, yaw]
        features.extend(head_pose)  
        

    # eye gaze direction (4 dims)
        #left_eye_center = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
        #                          for i in [33, 133]], axis=0)
        #right_eye_center = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
        #                           for i in [362, 263]], axis=0)
    # Compute gaze direction
        #left_gaze_x = (left_eye_center[0] - 0.3) * 2
        #left_gaze_y = (left_eye_center[1] - 0.4) * 2
        #right_gaze_x = (right_eye_center[0] - 0.7) * 2
        #right_gaze_y = (right_eye_center[1] - 0.4) * 2
        #features.extend([left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y])

    # FACS action units (simplified to 10 dims)
        aus = []
    # AU1 - inner brow raiser
        au1 = max(0, 0.5 - landmarks.landmark[55].y) * 10
        aus.append(au1)
    # AU2 - outer brow raiser
        au2 = max(0, 0.4 - landmarks.landmark[70].y) * 10
        aus.append(au2)
    # AU4 - brow lowerer / furrow
        brow_distance = abs(landmarks.landmark[55].x - landmarks.landmark[70].x)
        au4 = max(0, 0.1 - brow_distance) * 50
        aus.append(au4)
    # AU5 - upper eyelid raiser
        left_eye_open = abs(landmarks.landmark[33].y - landmarks.landmark[145].y)
        au5 = left_eye_open * 20
        aus.append(au5)
    # AU6 - cheek raiser
        cheek_height = landmarks.landmark[116].y
        au6 = max(0, 0.6 - cheek_height) * 15
        aus.append(au6)
    # AU9 - nose wrinkler
        nose_width = abs(landmarks.landmark[125].x - landmarks.landmark[141].x)
        au9 = max(0, nose_width - 0.02) * 100
        aus.append(au9)
    # AU12 - lip corner puller (smile)
        mouth_corner_avg = (landmarks.landmark[61].y + landmarks.landmark[84].y) / 2
        mouth_center = landmarks.landmark[13].y
        au12 = max(0, mouth_center - mouth_corner_avg) * 50
        aus.append(au12)
    # AU15 - lip corner depressor
        au15 = max(0, mouth_corner_avg - mouth_center) * 50
        aus.append(au15)
    # AU20 - lip stretcher (horizontal)
        mouth_width = abs(landmarks.landmark[61].x - landmarks.landmark[291].x)
        au20 = mouth_width * 100
        aus.append(au20)
    # AU25 - lips parting
        #mouth_open = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
        #au25 = mouth_open * 100
        #aus.append(au25)
        features.extend(aus) 


    # facial emotion features (6 dims)
        #emotions = []
    # Happiness - based on mouth corner lift
    #left_corner = landmarks.landmark[61].y
    #right_corner = landmarks.landmark[84].y
    #mouth_center = landmarks.landmark[13].y
    #corner_avg = (left_corner + right_corner) / 2
    #mouth_corner_lift = mouth_center - corner_avg
    #happiness = max(0, mouth_corner_lift) * 5
    #emotions.append(happiness)
        
    # Sadness - based on lip corner drop and brow drop
    #mouth_corner_drop = -min(0, mouth_corner_lift)
    #brow_drop = max(0, 0.45 - landmarks.landmark[55].y)
    #sadness = (mouth_corner_drop + brow_drop) * 3
    #emotions.append(sadness)
        
    # Anger - based on brow furrow
    #brow_furrow = max(0, 0.1 - abs(landmarks.landmark[55].x - landmarks.landmark[70].x))
    #anger = brow_furrow * 20
    #emotions.append(anger)
        
    # Disgust - based on nose scrunch
    #nose_scrunch = abs(landmarks.landmark[125].x - landmarks.landmark[141].x)
    #disgust = max(0, nose_scrunch - 0.02) * 50
    #emotions.append(disgust)
        
    # Surprise - based on brow raise and mouth opening
    #brow_raise = max(0, 0.4 - landmarks.landmark[70].y)
    #mouth_open = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
    #surprise = (brow_raise + mouth_open) * 10
    #emotions.append(surprise)
        
    # Fear - based on eye widen and brow raise
    #eye_wide = abs(landmarks.landmark[33].y - landmarks.landmark[145].y)
    #fear = (eye_wide + brow_raise) * 8
    #emotions.append(fear) 
    #features.extend(emotions) 

        features = features[:VISUAL_FEATURE_SIZE]
        while len(features) < VISUAL_FEATURE_SIZE:
            features.append(0.0)
        return np.array(features)


#-------------------------------------generate pkl files------------------------------------------------------
    def process_dataset(self, video_dir, csv_path, output_dir, audio_dir=None):
        """Process the entire dataset and generate .pkl files"""
        # If csv_path is None, auto-generate a CSV file
        if csv_path is None:
            csv_path = Path(video_dir).parent / "generated_dataset.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["video_id", "clip_id", "label", "split", "class_label"])  # write header
                for video_file in Path(video_dir).rglob("*.mp4"):
                    video_id = video_file.parent.name
                    clip_id = video_file.stem
                    writer.writerow([video_id, clip_id, -1, "test", "NEUT"])  # write default values
            print(f"Auto-generated CSV saved to: {csv_path}")

        # load label and dataset split information
        splits = {"train": [], "valid": [], "test": []}

        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:  # skip header
                video_id, clip_id, label, split, class_label = line.strip().split(",")
                video_path = Path(video_dir) / video_id / f"{clip_id}.mp4"
                splits[split].append((video_path, float(label), class_label))

    # initialize data structures
        data = {split: {"text": [], "language": [], "raw_text": [], "audio": [], "vision": [], "labels": [], "id": [], "class_labels": []} for split in splits}

    # Extract features
        for split, items in splits.items():
            for video_path, label, class_label in tqdm(items, desc=f"Processing {split} set"):
                if not video_path.exists():
                    print(f"⚠️ Video file not found: {video_path}")
                    continue

                # Find corresponding audio file
                audio_path = None
                if audio_dir:
                    audio_path = self.find_audio_file(video_path, audio_dir)

                # Extract features
                language, raw_text, text_features = self.extract_text_features(video_path, audio_path)
                audio_features = self.extract_audio_features(video_path, audio_path)
                visual_features = self.extract_visual_features(video_path)
                unique_id = f"{video_path.parent.name}_{video_path.stem}"

                # save features
                data[split]["text"].append(text_features)
                data[split]["language"].append(language)
                data[split]["raw_text"].append(raw_text)
                data[split]["audio"].append(audio_features)
                data[split]["vision"].append(visual_features)
                data[split]["labels"].append(label)
                data[split]["id"].append(unique_id)  # use unique ID
                data[split]["class_labels"].append(class_label)  # add class label

    # Save .pkl files
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            with open(output_dir / f"{split}_data.pkl", "wb") as f:
                pickle.dump(data[split], f)

    # Save metadata
        metadata = {
            "text_dim": TEXT_EMBEDDING_DIM,
            "audio_dim": AUDIO_FEATURE_SIZE,
            "visual_dim": VISUAL_FEATURE_SIZE,
            "num_classes": 5,
            "train_samples": len(data["train"]["labels"]),
            "val_samples": len(data["valid"]["labels"]),
            "test_samples": len(data["test"]["labels"]),
        }
        with open(output_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print("Processing completed and files saved successfully.")
#-------------------------------------Run main-----------------------------------------------------
if __name__ == "__main__":
    processor = MOSEIExtractor(language="zh")
    processor.process_dataset(
        video_dir="video2pkl/video2pkl/mix_video",
        csv_path="video2pkl/video2pkl/valid_video_zh.csv",
        output_dir="E:/kaggle/MSAbypkl/data/data_pkl/valid_zh_pkl",
        audio_dir=None
        #audio_dir="video2pkl/video2pkl/en_audio"
    )