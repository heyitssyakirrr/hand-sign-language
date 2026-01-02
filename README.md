# HealthSign AI 🏥✋

**HealthSign AI** is a real-time computer vision application designed to recognize specific health-related gestures and sign language.

Built with **Flask**, **MediaPipe**, and **TensorFlow**, it captures video input via webcam, extracts holistic keypoints (face, hands, pose), and classifies them into medical symptoms to assist in non-verbal communication for healthcare contexts.

## 🚀 Key Features

- **👁️ Real-Time Detection**: Instantly recognizes gestures via webcam streaming.
- **🧬 Holistic Tracking**: Uses **MediaPipe Holistic** to track face, pose, and hand landmarks simultaneously.
- **🏥 Symptom Recognition**: Currently trained to detect specific medical signs:
  - 🤕 **Headache**
  - 😷 **Coughing**
  - 🤒 **Sore Throat**
- **🌐 Web Interface**: A user-friendly web dashboard powered by Flask with authentication screens.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Computer Vision**: OpenCV (`cv2`), MediaPipe
- **Machine Learning**: TensorFlow / Keras (LSTM/Dense Neural Network)
- **Frontend**: HTML5, CSS3

## 🧠 How It Works

1.  **Data Collection**: Landmarks are extracted from video frames and stored as NumPy arrays (found in `MP_Data/`).
2.  **Training**: The `action.h5` model was trained on sequences of these landmarks to understand temporal movement.
3.  **Inference**:
    - The Flask app (`app.py`) streams video frames.
    - MediaPipe extracts keypoints for every frame.
    - The model predicts the probability of the gesture.
    - The result is rendered on the live feed.

## 📂 Project Structure

```bash
hand-sign-language/
├── MP_Data/          # Numpy arrays of collected landmark data
│   ├── coughing/
│   ├── headache/
│   └── sorethroat/
├── static/           # CSS, Images (logos), and compiled assets
├── templates/        # HTML pages (main.html, sign-in.html)
├── Logs/             # TensorBoard training logs
├── app.py            # Main Flask application entry point
├── action.h5         # Pre-trained Keras model
└── requirements.txt  # Python dependencies
