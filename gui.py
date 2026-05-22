import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import model_from_json


BASE_DIR = Path(__file__).resolve().parent
MODEL_JSON = BASE_DIR / "model_a1.json"
MODEL_WEIGHTS = BASE_DIR / "model_weights1.h5"
HAAR_CASCADE = BASE_DIR / "haarcascade_frontalface_default.xml"

# FER2013 correct order
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

INPUT_SIZE = (48, 48)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emotion-detector")


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_model():
    require_file(MODEL_JSON, "Model JSON")
    require_file(MODEL_WEIGHTS, "Model weights")

    with open(MODEL_JSON, "r") as f:
        model = model_from_json(f.read())

    model.load_weights(str(MODEL_WEIGHTS))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_cascade():
    require_file(HAAR_CASCADE, "Haar cascade")
    cascade = cv2.CascadeClassifier(str(HAAR_CASCADE))

    if cascade.empty():
        raise RuntimeError("Haar cascade not loaded")

    return cascade


def prepare_face(gray, face_box):
    x, y, w, h = face_box

    face = gray[y:y + h, x:x + w]

    # IMPORTANT: no equalizeHist, no extra padding
    face = cv2.resize(face, INPUT_SIZE)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    return face


def predict_emotion(gray, face_box):
    face = prepare_face(gray, face_box)

    predictions = model.predict(face, verbose=0)[0]

    print("\nPrediction Scores:")
    for name, score in zip(EMOTIONS_LIST, predictions):
        print(f"{name}: {score:.4f}")

    index = int(np.argmax(predictions))
    emotion = EMOTIONS_LIST[index]
    confidence = float(predictions[index])

    return emotion, confidence


class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")
        self.root.geometry("900x700")
        self.root.configure(background="#CDCDCD")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.camera = None
        self.camera_running = False
        self.photo_image = None

        self.build_ui()

    def build_ui(self):
        heading = tk.Label(
            self.root,
            text="Emotion Detector",
            pady=16,
            font=("Arial", 25, "bold"),
            background="#CDCDCD",
            foreground="#364156",
        )
        heading.pack()

        self.image_label = tk.Label(self.root, background="#CDCDCD")
        self.image_label.pack(side="top", expand=True, fill="both", padx=20, pady=10)

        self.result_label = tk.Label(
            self.root,
            text="Upload an image or start webcam",
            background="#CDCDCD",
            foreground="#011638",
            font=("Arial", 15, "bold"),
        )
        self.result_label.pack(side="top", pady=8)

        button_frame = tk.Frame(self.root, background="#CDCDCD")
        button_frame.pack(side="bottom", pady=28)

        upload_button = tk.Button(
            button_frame,
            text="Upload Image",
            command=self.upload_image,
            padx=12,
            pady=7,
            background="#364156",
            foreground="white",
            font=("Arial", 13, "bold"),
        )
        upload_button.grid(row=0, column=0, padx=8)

        self.webcam_button = tk.Button(
            button_frame,
            text="Start Webcam",
            command=self.toggle_webcam,
            padx=12,
            pady=7,
            background="#364156",
            foreground="white",
            font=("Arial", 13, "bold"),
        )
        self.webcam_button.grid(row=0, column=1, padx=8)

    def upload_image(self):
        self.stop_webcam()

        file_path = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        image = cv2.imread(file_path)

        if image is None:
            self.show_error("Image read panna mudiyala")
            return

        processed = self.detect_and_draw(image)
        self.show_frame(processed)

    def toggle_webcam(self):
        if self.camera_running:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.show_error("Webcam open aagala")
            return

        self.camera_running = True
        self.webcam_button.configure(text="Stop Webcam")
        self.update_webcam_frame()

    def stop_webcam(self):
        self.camera_running = False
        self.webcam_button.configure(text="Start Webcam")

        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def update_webcam_frame(self):
        if not self.camera_running:
            return

        ok, frame = self.camera.read()

        if not ok:
            self.stop_webcam()
            self.show_error("Webcam frame varala")
            return

        processed = self.detect_and_draw(frame)
        self.show_frame(processed)

        self.root.after(30, self.update_webcam_frame)

    def detect_and_draw(self, frame):
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
        )

        if len(faces) == 0:
            self.result_label.configure(text="No face detected")
            return display

        # largest face only
        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        x, y, w, h = faces[0]

        emotion, confidence = predict_emotion(gray, (x, y, w, h))

        label = f"{emotion} ({confidence:.0%})"

        cv2.rectangle(display, (x, y), (x + w, y + h), (36, 65, 86), 2)
        cv2.putText(
            display,
            label,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (36, 65, 86),
            2,
            cv2.LINE_AA,
        )

        self.result_label.configure(text=f"Predicted Emotion: {label}")

        return display

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((820, 500), Image.Resampling.LANCZOS)

        self.photo_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo_image)

    def show_error(self, message):
        self.result_label.configure(text=message)
        messagebox.showerror("Emotion Detector", message)

    def on_close(self):
        self.stop_webcam()
        self.root.destroy()


try:
    face_cascade = load_cascade()
    model = load_model()

except Exception as e:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Startup failed", str(e))
    raise


if __name__ == "__main__":
    app_root = tk.Tk()
    app = EmotionDetectorApp(app_root)
    app_root.mainloop()