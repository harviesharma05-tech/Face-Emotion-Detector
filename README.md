# 🎭 Face Emotion Detector

> Real-time facial emotion detection using OpenCV + DeepFace. Detects 7 emotions live from webcam with confidence scores and a visual bar chart overlay.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-000000?style=for-the-badge&logo=python&logoColor=white)

---

## 📸 What It Does

- Detects faces in real-time from webcam or any image/video
- Classifies 7 emotions: **Happy · Sad · Angry · Fear · Surprise · Disgust · Neutral**
- Shows confidence score + live probability bar chart per face
- Logs session data to CSV for analysis
- Generates emotion frequency stats chart

---

## 🗂️ Project Structure

```
face-emotion-detector/
│
├── emotion_detector.py       # Main script — run this
├── requirements.txt          # Dependencies
│
├── utils/
│   ├── logger.py             # Logs emotions to CSV
│   └── visualize_stats.py    # Plots session stats
│
└── output/                   # Screenshots, videos, logs saved here
```

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/harviesharma/face-emotion-detector.git
cd face-emotion-detector
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> First run downloads DeepFace models automatically (~500MB). Keep internet on.

---

## 🚀 Usage

### Webcam (default)
```bash
python emotion_detector.py
```

### Webcam — analyze every 3 frames (smoother)
```bash
python emotion_detector.py --skip 3
```

### Save output video
```bash
python emotion_detector.py --save
```

### Run on an image
```bash
python emotion_detector.py --source path/to/image.jpg
```

### Controls
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot to `output/` |

---

## 📊 Visualize Session Stats

After running a session:
```bash
python utils/visualize_stats.py
```
Generates `output/emotion_stats.png` — a bar chart of all detected emotions.

---

## 🧠 How It Works

```
Webcam Frame
    ↓
OpenCV — Face Detection (Haar Cascade / MTCNN)
    ↓
DeepFace — Deep Learning Model (VGG-Face / FER architecture)
    ↓
7-class Emotion Softmax Output
    ↓
Overlay on Frame (bounding box + bar chart)
    ↓
Optional: Log to CSV → Stats Plot
```

**Model:** DeepFace uses pre-trained FER (Facial Expression Recognition) models under the hood. No manual training needed.

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| OpenCV | Webcam capture, frame rendering |
| DeepFace | Pre-trained emotion model |
| TensorFlow | Deep learning backend |
| NumPy | Array operations |
| Matplotlib | Stats visualization |

---

## 🔮 Future Improvements

- [ ] Add Streamlit web UI
- [ ] Multi-face tracking with unique IDs
- [ ] Emotion timeline graph (live plot)
- [ ] Export to PDF report
- [ ] Edge deployment with ONNX

---

## 👨‍💻 Author

**Harvi Sharma**  
B.Tech CSE @ Graphic Era University (2025–2029)  
[![GitHub](https://img.shields.io/badge/GitHub-harviesharma-181717?style=flat-square&logo=github)](https://github.com/harviesharma)

---

## 📄 License

MIT License — free to use, modify, and distribute.
