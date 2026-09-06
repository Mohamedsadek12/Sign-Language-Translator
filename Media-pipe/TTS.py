import cv2
import os
import json
import time
import threading
import pygame
import numpy as np

from gtts import gTTS
from tensorflow.keras.models import load_model

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)


MODEL_PATH = "asl_landmark_model.h5"
CLASSES_PATH = "landmark_classes.json"
HAND_TASK_PATH = "hand_landmarker.task"
HOLD_TIME = 1.5
CONFIDENCE_THRESHOLD = 70


model = load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    idx_to_class = json.load(f)

idx_to_class = { int(k): v for k, v in idx_to_class.items() }

print("Model loaded successfully")
print("Classes loaded successfully")


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_TASK_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)



def extract_landmarks(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None, result

    hand = result.hand_landmarks[0]

    coords = np.array([[p.x, p.y] for p in hand], dtype=np.float32)

    # Wrist = landmark 0
    wrist = coords[0].copy()

    # Move wrist to origin
    coords -= wrist

    # Middle finger MCP = landmark 9
    scale = np.linalg.norm(coords[9])

    if scale < 1e-6:
        return None, result

    # Scale normalization
    coords /= scale

    # 21 × 2 = 42
    return coords.flatten(), result


# Draw MediaPipe landmarks
def draw_landmarks(frame, result):

    if not result.hand_landmarks:
        return

    h, w = frame.shape[:2]

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    for hand in result.hand_landmarks:

        points = []

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            points.append((x, y))

        # Draw connections
        for start, end in connections:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

        # Draw points
        for x, y in points:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


# TTS
pygame.mixer.init()

def speak(text):

    text = text.strip().lower()

    if not text:
        return

    print(f"🔊 Speaking: '{text}'")

    filename = f"speech_{int(time.time() * 1000)}.mp3"

    try:
        print("Generating speech...")

        tts = gTTS(text=text, lang="en", slow=False)

        tts.save(filename)

        print("Playing speech...")

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()

        pygame.mixer.music.unload()

        print("Speech finished.")
        
    except Exception as e:
        print(f"TTS error: {e}")

    finally:

        if os.path.exists(filename):
            try:
                os.remove(filename)
                print("Temporary speech file deleted.")
            except Exception as e:
                print(f"Error deleting temporary speech file: {e}")


# Wepcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")

    landmarker.close()
    pygame.mixer.quit()

    exit()


print("\nWebcam started")
print()
print("Controls:")
print("S = Speak ASL sentence")
print("C = Clear")
print("Q = Quit")



# Text State
current_word = ""
sentence = ""
last_label = ""
last_label_time = time.time()
letter_added = False


# Main Loop
while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]


    # Extract landmarks
    landmarks, result = extract_landmarks(frame)

    # Draw MediaPipe landmarks
    draw_landmarks(frame, result)

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    label = "No hand"
    confidence = 0.0

    # Predict
    if landmarks is not None:
        input_data = np.expand_dims(landmarks, axis=0)
        prediction = model.predict(input_data, verbose=0)[0]
        class_index = np.argmax(prediction)
        label = idx_to_class[class_index]
        confidence = (prediction[class_index] * 100)

    # Letter Detection
    if label != "No hand":
        if label == last_label:
            elapsed = (time.time() - last_label_time)

            if (elapsed >= HOLD_TIME and not letter_added and confidence >= CONFIDENCE_THRESHOLD):

                # Space
                if label == "space":
                    if current_word:
                        sentence += (current_word + " ")
                        print(f"Word completed: {current_word}")
                        current_word = ""


                # Delete
                elif label == "del":
                    current_word = (current_word[:-1])


                # Nothing
                elif label == "nothing":
                    pass


                # Letter
                else:
                    current_word += label
                    print(f"Letter: {label}")


                letter_added = True

        else:
            last_label = label
            last_label_time = time.time()
            letter_added = False


    # UI
    if confidence >= CONFIDENCE_THRESHOLD:
        box_color = (0, 255, 0)

    else:
        box_color = (0, 165, 255)


    # Prediction
    cv2.putText(frame, f"{label} ({confidence:.1f}%)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 2)

    # Current word
    cv2.putText(frame,f"Word: {current_word}", (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Sentence
    cv2.putText(frame, f"Sentence: {sentence}", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


    # Controls
    cv2.putText(frame, "S: Speak | C: Clear | Q: Quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


    # Hold Progress
    if ( label == last_label and confidence >= CONFIDENCE_THRESHOLD):

        elapsed = min(time.time() - last_label_time, HOLD_TIME)
        progress = int((elapsed / HOLD_TIME) * 300)

        cv2.rectangle(frame, (20, 100), (20 + progress, 120), (0, 255, 0), -1)

    cv2.rectangle(frame, (20, 100), (320, 120), (255, 255, 255), 1)


    # Display

    cv2.imshow("ASL Translator", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit

    if key == ord("q"):
        break


    # TTS
    elif key == ord("s"):

        # Add unfinished word
        if current_word:
            sentence += (current_word + " ")
            current_word = ""

        text = sentence.strip()

        if text:
            speak(text)

        else:
            print("Nothing to speak.")


    # Clear
    elif key == ord("c"):

        current_word = ""
        sentence = ""
        last_label = ""
        letter_added = False
        print("🗑️ Cleared")

cap.release()
cv2.destroyAllWindows()
landmarker.close()
pygame.mixer.quit()