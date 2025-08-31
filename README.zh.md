# 🌏 跨语言多模态情感分析 (MSA)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

> **中英跨语言多模态情感分析框架**
> 🎥 端到端 **视频 → 情感** 预测
> 🔗 基于 Transformer 的 **跨注意力融合**
> 🔍 内置 **可解释性与可视化**
> 🏆 为第 4 届琶洲 AI 大赛 MSA 赛道开发

---

## ✨ 主要特性

- 🌐 **跨语言**：支持中文和英语，文本使用 **Whisper + BERT**。
- 🎬 **多模态**：
  - **文本** → Whisper 转录 + BERT 嵌入
  - **音频** → MFCC + prosody 特征
  - **视觉** → MediaPipe 关键点 + HoG + FACS
- 🔗 **融合**：基于 Transformer 的 **跨注意力**，保证模态间平衡贡献。
- 🔍 **可解释性**：注意力热力图展示各模态贡献。
- ⚡ **灵活运行**：
  - 一键演示（`test_script.py`）
  - 两步可复现流程（视频 → 特征 → 训练）
  - 直接视频预测（不保存 `.pkl`）
- 🧩 **模块化设计**：便于接入 CLIP、SAM、LLM 等扩展。

---

## 📂 项目结构

```bash
MSA/
├── README.md                # 项目介绍（入口）
├── requirements.txt         # 依赖
├── test_script.py           # 一键测试脚本
├── best_models/             # 预训练/最佳模型（en.pt / zh.pt）
├── MSAbypkl/                # 基于 PKL 的训练/评估流程
│   ├── main.py	     # 训练与评估
|   ├── config.py            # 参数配置
│   ├── scripts/             # 训练脚本
│   └── src/                 # 核心模块（data/models/training/utils）
├── MSAbyvideo/              # 直接视频到情感的工作流
├── video2pkl/               # 视频到特征提取器
│   └── video2pkl.py         # Whisper + BERT + MediaPipe 特征提取
└── Test_Data / Test_Results # 示例数据与输出
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 一键运行（推荐）

```bash
python test_script.py
```

- 将视频放到 `Video_Data/`
- 预测结果保存在 `Test_Results/`

### 3️⃣ 两步工作流

**步骤 1：提取特征**

```bash
python video2pkl/video2pkl/video2pkl.py     --language zh     --video_dir ./Video_Data     --csv_path ./meta.csv     --output_dir ./MSAbypkl/data/data_pkl/myset
```

**步骤 2：训练 / 评估**

```python
# config.py
DATASET_NAME = "myset"
```

```bash
python MSAbypkl/main.py
```

### 4️⃣ 直接视频到情感

```bash
python MSAbyvideo/main.py
```

- 交互式命令行提示
- 更快，不保存 `.pkl`

---

## 🧠 模型设计

- **文本**：Whisper → BERT（768 维）
- **音频**：MFCC + prosody（40 维）
- **视觉**：MediaPipe + HoG + FACS（35 维）
- **融合**：Transformer 编码器 + 跨注意力
- **输出**：5 类情感 → `SNEG | WNEG | NEUT | WPOS | SPOS`
  *(回归支持 `NUM_CLASSES=1`)*

📊 **评估指标**：准确率、Macro-F1、混淆矩阵

---

## 🔧 常见问题与排查

以下是我们本地调试时遇到的问题、典型原因及建议修复方式，帮助你快速复现结果。

### 1）FileNotFoundError: ffmpeg 未找到

- 现象：调用 `Whisper.transcribe()` 或 MoviePy 打开视频时（底层调用 `ffmpeg` 可执行文件）抛出异常。
- 原因：系统中没有安装 `ffmpeg` 或未加入 PATH，子进程找不到该可执行文件。
- Windows 下修复示例：
  - 使用 winget（推荐）：
    ```powershell
    winget install --id Gyan.FFmpeg -e --silent
    ```
  - 使用 Chocolatey（需要管理员）：
    ```powershell
    choco install -y ffmpeg
    ```
  - 手动安装：下载 ffmpeg 并把其 `bin` 目录加入 PATH（例如 `C:\ffmpeg\bin`）：
    ```powershell
    setx PATH "$env:Path;C:\ffmpeg\bin"
    ```
  - 验证安装：
    ```powershell
    ffmpeg -version
    ```

注意：我们也在代码层面做了小修复，避免 MoviePy 在 WindowsPath 下出问题——确保以字符串形式传递文件路径（参见 `video2pkl/video2pkl.py` 和 `video2pkl/video2csd/get_3m.py`）。

关于回退行为的说明：

- 当系统 `ffmpeg` 不可用时，代码会尝试回退：如果安装了 `imageio-ffmpeg`，项目会定位其捆绑的 ffmpeg 二进制，并临时复制到名为 `msaffmpeg` 的本地临时目录，再把该目录 prepend 到进程 `PATH` 中，使子进程可调用。
- 因此在没有系统级 ffmpeg 的情况下，只要安装了 `imageio-ffmpeg`，`test_script.py` 和 Whisper/MoviePy 仍能工作。`requirements.txt` 中已包含 `imageio-ffmpeg==0.5.1`。

### 2）FileNotFoundError: best_models/zh.pt（或 en.pt）

- 现象：调用 `torch.load(...)` 加载 checkpoint 时会抛出 FileNotFoundError（例如在 `MSAbypkl/main.py` 中评估阶段）。
- 修复选项：
  - 下载所需 checkpoint 文件并放到仓库根目录的 `best_models/` 文件夹，且文件名与代码中期待的相同（例如 `best_models/zh.pt`, `best_models/en.pt`）。
  - 或者在 `MSAbypkl/main.py` 中把 checkpoint 路径改为你本地的路径。

### 3）ImportError: recursion is detected during loading of "cv2" binary extensions

- 现象：Python 中导入 cv2 时 OpenCV 加载失败，出现递归错误。
- 原因：OpenCV 安装冲突或安装损坏，通常由于多个版本或安装不完整导致。
- 修复方法：
  - 卸载现有的 OpenCV 包：
    ```bash
    pip uninstall opencv-contrib-python opencv-python -y
    ```
  - 重新安装稳定版本：
    ```bash
    pip install opencv-python==4.8.0.76
    ```
  - 验证安装：
    ```bash
    python -c "import cv2; print('OpenCV version:', cv2.__version__)"
    ```

### 4）RuntimeError: Model not found（Whisper 模型加载错误）

- 现象：Whisper 模型加载失败，出现路径相关错误。
- 原因：代码中硬编码的模型路径在当前系统上不存在。
- 修复方法：代码已更新为使用在线 Whisper 模型而不是本地路径。确保有网络连接进行初始模型下载，或者根据需要更新代码以使用本地模型路径。

---

## 📥 预训练模型与最佳模型

- 预训练的文本模型与我们的最佳 checkpoint（中/英）可以在以下位置获得：

  https://drive.google.com/drive/folders/1deCsD3TXacpuov78v7PldhXL5zFSjNjL
- 下载后放到仓库根目录的 `best_models/` 文件夹中：

  - `best_models/zh.pt`
  - `best_models/en.pt`
  - 或者在 `MSAbypkl/main.py` 中修改加载路径为你的自定义位置。

---

## 📊 运行结果

- 🏆 **公共排行榜**：`0.4350`（基线水平，CPU-only 训练）
- 🔬 **消融实验**：跨注意力 > 早期融合 > 迟钝融合
- ⚡ 使用 GPU 时，准确率有望显著提升

---

## 🔍 可解释性

- 保存 **跨注意力权重**
- 生成 **模态贡献热力图**
- 提供透明且可信的预测解释

---

## 📬 联系方式

👨‍💻 **团队成员**：YanSong Hu · YangLe Ma · ZhouYang Wang · RiJie Hao
📧 邮箱：672416680@qq.com | mylsxxy@163.com

⭐ 如果你喜欢本项目，请给仓库点个 Star！ ⭐
