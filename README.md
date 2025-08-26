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

## 🔧 Troubleshooting / Common Issues

Below are the issues we encountered during local debugging, their typical causes, and recommended fixes to help you reproduce results quickly.

### 1) FileNotFoundError: ffmpeg not found

- Symptom (example trace): raised when calling `Whisper.transcribe()` or when MoviePy opens a video file (under the hood these call the `ffmpeg` executable).
- Cause: Whisper and MoviePy rely on the system `ffmpeg` binary. If `ffmpeg` is not installed or not on PATH, a subprocess FileNotFoundError is raised.
- Fix (Windows examples):
  - Using winget (recommended if available):
    ```powershell
    winget install --id Gyan.FFmpeg -e --silent
    ```
  - Using Chocolatey (requires admin):
    ```powershell
    choco install -y ffmpeg
    ```
  - Manual install: download ffmpeg and add its `bin` folder to PATH (e.g. `C:\ffmpeg\bin`):
    ```powershell
    setx PATH "$env:Path;C:\ffmpeg\bin"
    ```
  - Verify installation:
    ```powershell
    ffmpeg -version
    ```

Note: We also applied a small code-level fix to avoid a MoviePy Path/WindowsPath issue by ensuring file paths are passed as strings (see `video2pkl/video2pkl.py` and `video2pkl/video2csd/get_3m.py`).

Note about fallback behavior:

- The codebase now attempts a graceful fallback when a system `ffmpeg` is not available: if the Python package `imageio-ffmpeg` is installed, the project will locate its bundled ffmpeg binary and make it available to subprocess calls by temporarily copying it to a local temp directory named `msaffmpeg` and prepending that directory to the process `PATH`.
- This makes `test_script.py` and Whisper/MoviePy work even when ffmpeg is not installed system-wide. The fallback relies on `imageio-ffmpeg` being installed; `imageio-ffmpeg==0.5.1` is already included in `requirements.txt`.

### 2) FileNotFoundError: best_models/zh.pt (or en.pt)

- Symptom: a FileNotFoundError occurs when `torch.load(...)` is called to load a checkpoint (this can appear during evaluation in `MSAbypkl/main.py`).
- Fix options:
  - Download the required checkpoint files and place them into the repository `best_models/` folder using the exact filenames (e.g. `best_models/zh.pt`, `best_models/en.pt`).
  - Or update the checkpoint paths in `MSAbypkl/main.py` to point to your local checkpoint locations.

---

## 📥 Pre-trained & Best Models

- Pre-trained text models and our best checkpoints (Chinese / English) are available here:

  https://drive.google.com/drive/folders/1deCsD3TXacpuov78v7PldhXL5zFSjNjL
- After downloading, place the files under the repository root `best_models/` directory:

  - `best_models/zh.pt`
  - `best_models/en.pt`
  - Alternatively, update `MSAbypkl/main.py` to load from your custom paths.

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
