# Real-Time-Emotional-Detector-Application

A Python GUI application that detects human emotions from an uploaded image or webcam using OpenCV, Tkinter, and a trained Keras model.

## Features

- Upload image and detect emotion
- Real-time webcam emotion detection
- Face detection using Haar Cascade
- Emotion prediction using trained CNN model
- Simple Tkinter GUI

## Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Project Files

```text
project-folder/
│── gui.py
│── model_a1.json
│── model_weights1.h5
│── haarcascade_frontalface_default.xml
```

## Requirements

Install the required packages:

```bash
pip install opencv-python numpy pillow tensorflow
```

## How to Run

1. Keep all files in the same folder.
2. Open terminal in the project folder.
3. Run:

```bash
python gui.py
```

## Usage

### Upload Image

Click **Upload Image** and select a face image.

### Webcam

Click **Start Webcam** to detect emotion live using camera.

## Model Files Required

The app needs these files in the same folder:

```text
model_a1.json
model_weights1.h5
haarcascade_frontalface_default.xml
```

## Main Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- Tkinter
- Pillow
- NumPy

## Author

**Manoj Raj G**
