# 🌏 Cross-Lingual Multimodal Sentiment Analysis (MSA)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> **Chinese–English Cross-Lingual Multimodal Sentiment Analysis Framework**
> 🎥 End-to-end **Video → Sentiment** prediction
> 🔗 Transformer-based **Cross-Attention Fusion**
> 🔍 Built-in **Explainability & Visualization**
> 🏆 Developed for **MSA Challenge @ The 4th Pazhou AI Competition**

---

## ✨ Key Features

- 🌐 **Cross-Lingual**: Handles both **Chinese** and **English** with **Whisper + BERT**.
- 🎬 **Multimodal**:
  - **Text** → Whisper transcript + BERT embeddings
  - **Audio** → MFCC + prosody features
  - **Visual** → MediaPipe landmarks + HoG + FACS
- 🔗 **Fusion**: Transformer-based **Cross-Attention** for balanced modality contribution.
- 🔍 **Explainability**: Attention heatmaps reveal modality importance.
- ⚡ **Flexible Execution**:
  - One-click demo (test_script.py)
  - Two-step reproducible pipeline (video → features → training)
  - Direct video-to-prediction (no `.pkl` saved)
- 🧩 **Modular Design**: Easily extend with CLIP, SAM, LLMs, etc.

---

## 📂 Project Structure

```bash
MSA/
├── README.md                # Project introduction
├── requirements.txt         # Dependencies
├── test_script.py           # One-click testing
├── best_models/             # Pre-trained weights (en.pt / zh.pt)
├── MSAbypkl/                # PKL-based workflow (train/eval)
│   ├── main.py		     # Training & evaluation
|   ├── config.py            # Modify parameters
│   ├── scripts/             # Training scripts
│   └── src/                 # Core modules (data/models/training/utils)
├── MSAbyvideo/              # Direct video-to-sentiment workflow
├── video2pkl/               # Video-to-feature extractor
│   └── video2pkl.py         # Whisper + BERT + MediaPipe extractor
└── Test_Data / Test_Results # Example data & outputs
```

📖 More details: 

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ One-Click Execution (Recommended)

```bash
python test_script.py
```

- Put videos into `Video_Data/`
- Predictions saved in `Test_Results/`

### 3️⃣ Two-Step Workflow

**Step 1: Extract Features**

```bash
python video2pkl/video2pkl/video2pkl.py     --language en     --video_dir ./Video_Data     --csv_path ./meta.csv     --output_dir ./MSAbypkl/data/data_pkl/myset
```

**Step 2: Train / Evaluate**

```python
# config.py
DATASET_NAME = "myset"
```

```bash
python MSAbypkl/main.py
```

### 4️⃣ Direct Video-to-Sentiment

```bash
python MSAbyvideo/main.py
```

- Interactive CLI prompts
- Faster, no `.pkl` saved

---

## 🧠 Model Design

- **Text**: Whisper → BERT (768-dim)
- **Audio**: MFCC + prosody (40-dim)
- **Visual**: MediaPipe + HoG + FACS (35-dim)
- **Fusion**: Transformer encoders + Cross-Attention
- **Output**: 5-class sentiment → `SNEG | WNEG | NEUT | WPOS | SPOS`
  *(Regression supported via `NUM_CLASSES=1`)*

📊 **Metrics**: Accuracy, Macro-F1, Confusion Matrix

---

## 📊 Results

- 🏆 **Public Leaderboard**: `0.4350` (baseline-level, CPU-only training)
- 🔬 **Ablations**: Cross-attention > Early fusion > Late fusion
- ⚡ With GPU, accuracy expected to improve significantly

---

## 🔍 Explainability

- Saves **cross-attention weights**
- Generates **modality contribution heatmaps**
- Ensures **transparent & trustworthy predictions**

---

## 📬 Contact

👨‍💻 **Team Members**: YanSong Hu · YangLe Ma · ZhouYang Wang · RiJie Hao
📧 Email: 672416680@qq.com | mylsxxy@163.com

⭐ If you like this project, please **star the repo** — it keeps us motivated! ⭐
