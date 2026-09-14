# 🤟 Sign Language Translator

A computer vision and deep learning project designed to translate sign language into understandable communication in real time.

The project is being developed in multiple stages, with the current implementation focusing on **ASL alphabet recognition** as the foundation of the system.

---
## 👥 Team

- **Mohamed Sadek** — [GitHub](https://github.com/Mohamedsadek12)
- **Zyad Salah** — [GitHub](https://github.com/zyad-elkhewekh)

---
## 🎯 Current Stage

The current stage focuses on building a reliable **real-time ASL recognition component** capable of identifying individual alphabet signs from a webcam.

Rather than relying on a single approach, we experimented with different computer vision and deep learning techniques to compare their performance and suitability for real-time use.

---

## 🧠 Approaches

### 1. Custom CNN

We started by developing a **Convolutional Neural Network (CNN)**.

The model learns directly from hand images and automatically extracts visual features such as:

- Edges
- Shapes
- Hand contours
- Finger configurations
- Spatial patterns

This provided a baseline for image-based sign recognition and helped us understand how CNNs perform on the ASL recognition task.

---

### 2. MobileNetV2

We then experimented with **MobileNetV2 using transfer learning**.

MobileNetV2 was selected because it provides a strong balance between:

- Feature extraction capability
- Model size
- Computational efficiency
- Real-time inference performance

We used an ImageNet-pretrained MobileNetV2 and fine-tuned it for our sign recognition dataset.

This allowed us to leverage previously learned visual representations while adapting the model to the specific characteristics of sign language.

---
### 3. MediaPipe Hand Landmarks + Neural Network

Our third approach used a different representation of the input.

Instead of feeding the complete image directly into the classifier, we used **MediaPipe Hand Landmarker** to extract the structure of the hand.

MediaPipe provides **21 landmarks** for a detected hand. We extracted the normalized `(x, y)` coordinates, resulting in:

```text
21 landmarks × 2 coordinates = 42 features
```

The coordinates were normalized relative to the wrist and scaled according to the hand geometry before being passed to the classifier.

This approach focuses on the **geometric structure and configuration of the hand** rather than depending entirely on raw image pixels.

---

## 📊 Results

We compared the different approaches based on their validation performance.

| Approach | Validation Accuracy |
|----------|---------------------|
| Custom CNN | TBD |
| MobileNetV2 | TBD |
| MediaPipe Landmarks + Neural Network | **89.30%** |

The **MediaPipe landmark-based approach achieved our highest validation accuracy of 89.30%** in our experiments.

It also provides a compact representation of the hand, making it well suited to our real-time recognition pipeline.

> The CNN and MobileNetV2 accuracy values will be added once the final experiments are completed.

---
## 📷 Real-Time Recognition

The current system uses a webcam to detect and classify ASL hand signs in real time.

The recognition pipeline is:

```text
             Webcam
                │
                ▼
        Hand Detection
                │
                ▼
       21 Hand Landmarks
                │
                ▼
     Coordinate Normalization
                │
                ▼
      Neural Network Classifier
                │
                ▼
        Predicted ASL Letter
```

The application provides real-time information including:

- Detected hand landmarks
- Predicted ASL letter
- Prediction confidence
- Recognized letters

---
## 🔊 Text-to-Speech

The real-time implementation also includes a **Text-to-Speech (TTS)** component.

Recognized output can be converted into spoken audio using:

- **gTTS** for speech generation
- **Pygame** for audio playback

The TTS implementation is located in:

```text
Media-pipe/
└── TTS.py
```

---
## 📁 Project Structure

```text
Sign-Language-Translator/
│
├── ArSL_Media-pipe/
│   ├── arsl_landmark_classes.json
│   ├── arsl_landmark_model.h5
│   ├── arsl_mediapipe_model.py
│   ├── arsl_test.py
│   └── hand_landmarker.task
│
├── Media-pipe/
│   ├── TTS.py
│   ├── asl_landmark_model.h5
│   ├── hand_landmarker.task
│   ├── landmark_classes.json
│   ├── mediapipe_landmark_model.py
│   └── test_mediapipe.py
│
├── MobileNet/
│   ├── MobileNet_Model.py
│   ├── best_asl_mobilenet.h5
│   ├── best_asl_mobilenet_p1.h5
│   ├── class_indices.json
│   └── test_mobilenet_model.py
│
├── asl_custom_cnn/
│   ├── SLT.py
│   ├── asl_custom_cnn.h5
│   ├── asl_custom_cnn.keras
│   ├── class_indices.json
│   ├── real-time_test.py
│   ├── test_results.png
│   └── training_curves.png
│
├── .gitignore
├── ASL_to_voice
├── README.md
└── report.md
```

---
## 🛠️ Technologies

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- CNN
- MobileNet
- Deep Learning
- Computer Vision
- gTTS
- Pygame

---
## ⚙️ Installation

Create a virtual environment:

```bash
python -m venv slt-env
```

Activate it on Windows:

```bash
slt-env\Scripts\activate
```

Install the required dependencies:

```bash
pip install tensorflow opencv-python mediapipe numpy gTTS pygame
```

---
## 🚀 Running the Real-Time System

Make sure the required model, class mapping, and MediaPipe task files are available.

For the MediaPipe real-time implementation:

```bash
python Media-pipe/TTS.py
```

The webcam will open and the system will begin detecting hand landmarks and recognizing ASL signs.

---
## 🎮 Controls

| Key | Action |
|-----|--------|
| `S` | Speak recognized text |
| `C` | Clear recognized text |
| `Q` | Quit application |

---
## 📚 Key Takeaways

This stage provided practical experience with:

- Convolutional Neural Networks
- Transfer Learning / Lightweight CNN architectures
- Hand Landmark Detection
- Feature Engineering
- Coordinate Normalization
- Model Evaluation
- Real-Time Inference
- Computer Vision Pipelines
- Text-to-Speech Integration

One of the main insights from the experiments was that **the way the input is represented can be just as important as the model architecture itself**.

Rather than relying only on increasingly complex image models, representing the hand through its geometric landmarks provided a compact and effective input for real-time recognition.

---
## 🤝 Contributors

### Mohamed Sadek 
[GitHub](https://github.com/Mohamedsadek12) · [LinkedIn](https://www.linkedin.com/in/mohamed-sadek12/)

### Zyad Salah

[GitHub](https://github.com/zyad-elkhewekh) · [LinkedIn](https://www.linkedin.com/in/zyad-salah-79b731216/)

---

## ⭐ Project

This is an ongoing project focused on developing a practical, real-time **Sign Language Translator** using computer vision and deep learning.


