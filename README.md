# 🎯 Face Recognition with Python

A simple real-time face recognition system using `face_recognition` and `OpenCV`. Capture images, register known faces, and recognize them live from a webcam feed.

---

## 📦 Setup Instructions

### 1. Create and Activate a Virtual Environment

```sh
python -m venv face-env
.\face-env\Scripts\activate
```

---

### 2. Install Dependencies

Once your virtual environment is activated, install the required packages:

### Install Visual Studio Build Tools

Download and install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

During installation, check:

- C++ CMake Tools for Windows
- MSVC v142 (or latest version)
- Windows 10/11 SDK
- C++ Development Workload

### Install CMake

Download and install [CMake](https://cmake.org/download/) and add it to the system PATH.

Verify the installation:

```sh
cmake --version
```

### Install Python Packages

```sh
pip install dlib
pip install opencv-python
pip install face_recognition
pip install distribute setuptools
```

### Clone and Install Face Recognition Models

```sh
git clone https://github.com/ageitgey/face_recognition_models.git
cd face_recognition_models
pip install .
```

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

- **Detect faces and draw a rectangle:**

  ```sh
  python rectangle.py
  ```

- **Detect faces and draw facial landmarks:**
  ```sh
  python landmarks.py
  ```

Press `Q` to exit the video feed.

---
