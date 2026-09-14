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

## 🛠️ Technologies

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **MediaPipe**
- **NumPy**
- **MobileNetV2**
- **Deep Learning**
- **Computer Vision**

---



