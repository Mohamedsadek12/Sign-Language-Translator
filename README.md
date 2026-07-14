# ASL Sign Language Translator

A real-time American Sign Language (ASL) translator that converts hand signs to letters, builds words, and speaks them aloud using text-to-speech.

---

## Demo

```
Webcam → Hand Sign → Letter → Word → 🔊 Speech
```

Show a hand sign in the green box → hold for 1.5 seconds → letter is added → spell a word → press S to speak it.

---

## Features

- 🤟 Real-time ASL hand sign recognition via webcam
- 🔤 Spell words letter by letter using ASL alphabet
- 🔊 Text-to-speech output using Google TTS
- ✋ Supports all 26 letters + `space`, `del`, `nothing`
- 📊 Confidence indicator and hold progress bar
- ⌨️ Keyboard controls for speak, clear, and quit

---

## Dataset

**ASL Alphabet Dataset** from Kaggle:
```
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
```

- 87,000 images total
- 29 classes (A–Z + del, space, nothing)
- 3,000 images per class
- Image size: 200×200 pixels (resized to 128×128 for training)

---

## Model Architecture

### MobileNetV2 (Transfer Learning) — Final Model

```
Input (128×128×3)
        ↓
MobileNetV2 base (pretrained on ImageNet)
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(256, relu)
        ↓
Dropout(0.5)
        ↓
Dense(128, relu)
        ↓
Dropout(0.3)
        ↓
Dense(29, softmax)
```

### Training Strategy — Two Phases

| Phase | Base Model | Learning Rate | Epochs |
|---|---|---|---|
| Phase 1 | Frozen | 0.001 | 10 |
| Phase 2 | Last 30 layers unfrozen | 0.00001 | 10 |

### Results

| Metric | Value |
|---|---|
| Validation Accuracy | ~90% |
| Training Platform | Kaggle (2× NVIDIA Tesla T4) |

---

## Installation

**1 — Clone the repository**
```bash
git clone https://github.com/yourusername/sign-language-translator.git
cd sign-language-translator
```

**2 — Create a virtual environment**
```bash
python -m venv slt-env
slt-env\Scripts\activate        # Windows
source slt-env/bin/activate     # Linux/Mac
```

**3 — Install dependencies**
```bash
pip install tensorflow==2.12.0
pip install opencv-python matplotlib scipy keras pandas
pip install gTTS pygame
```

---

## Requirements

```
Python        3.10
TensorFlow    2.12.0
Keras         2.12.0
OpenCV        4.7.0+
NumPy         1.23.5
gTTS          2.3+
pygame        2.5+
scipy         1.10+
matplotlib    3.7+
```

---

## Usage

### Train the Model
```bash
python MobileNet_Model.py
```
Trains MobileNetV2 in two phases and saves:
- `best_asl_mobilenet.h5` — best model weights
- `class_indices.json` — class mapping

### Test on Static Images
```bash
python test_mobilenet_model.py
```
Runs predictions on all images in `dataset/asl_alphabet_test/` and shows results.

### Real-time Webcam Translator
```bash
python ASL_to_voice.py
```

---

## Controls

| Key | Action |
|---|---|
| Hold sign 1.5s | Add letter to current word |
| `space` sign | Complete word, add to sentence |
| `del` sign | Delete last letter |
| **S** | Speak the current sentence |
| **C** | Clear word and sentence |
| **Q** | Quit |

---

## How It Works

```
1. Webcam captures frame
2. Region of interest (ROI) — green box in center — is cropped
3. ROI is resized to 128×128 and preprocessed with MobileNetV2's preprocess_input
4. Model predicts the ASL letter with confidence score
5. If the same sign is held for 1.5 seconds with >70% confidence → letter registered
6. Letters build into words, words build into a sentence
7. Press S → sentence converted to speech via Google TTS
```

---

## Tips for Best Accuracy

- Keep your hand **inside the green box**
- Use a **plain background** behind your hand
- Ensure **good lighting** — avoid shadows on your hand
- Hold the sign **still** for 1.5 seconds until the progress bar fills
- The box turns **orange** when confidence is below 70% — adjust your hand position

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Model not found | Make sure `best_asl_mobilenet.h5` is in the same folder as the script |
| Webcam not opening | Check camera index: try `cv2.VideoCapture(1)` instead of `0` |
| TTS not working | Check internet connection — gTTS requires internet |
| Letters spoken individually | Text is uppercase — `.lower()` is applied automatically |
| Low accuracy on webcam | Improve lighting and use plain background |

---
