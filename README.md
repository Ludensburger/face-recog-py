# 🎯 Face Recognition with Python

A simple real-time face recognition system using `face_recognition` and `OpenCV`. Capture images, register known faces, and recognize them live from a webcam feed.

---

## 📦 Setup Instructions

### 1. Create and Activate a Virtual Environment

```sh
python -m venv face-env
.\face-env\Scripts\activate  # On Windows
```

---

### 2. Install Dependencies

Once your virtual environment is activated, install the required packages:

```sh
pip install cmake
pip install dlib

pip install face_recognition

git clone https://github.com/ageitgey/face_recognition_models.git
cd face_recognition_models
pip install .

pip3 install distribute
pip install setuptools
```

> ⚠️ Note: You may also need to install `cmake` and `dlib` manually if you encounter build issues.

---

## 🖼️ Register Known Faces

Place your photos in the following folder structure:

```
photos/
├── Person1/
│   ├── img1.jpg
│   └── img2.jpg
├── Person2/
│   ├── angle1.jpg
│   └── angle2.jpg
```

To capture and save images for face registration, run:

```sh
python register-face.py
```

---

## 🧠 Run Face Recognition

To start real-time face recognition:

```sh
python TESTING.py
```

Press `Q` to exit the video feed.

---
