from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import threading

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Global variables
eye_position = None
calibration_data = {}

# Lock for thread-safe operations
lock = threading.Lock()

def generate_frames():
    global eye_position
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks
                left_eye = [33, 133]
                right_eye = [362, 263]

                h, w, _ = frame.shape
                left = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye]
                right = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye]

                # Calculate center of eyes
                left_center = np.mean(left, axis=0)
                right_center = np.mean(right, axis=0)
                center = np.mean([left_center, right_center], axis=0)

                with lock:
                    eye_position = {'x': center[0], 'y': center[1]}

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict_gaze')
def predict_gaze():
    with lock:
        if not calibration_data or not eye_position:
            return jsonify(x=-1, y=-1)

        try:
            eye = np.array(list(eye_position.values()))
            center = np.array(list(calibration_data['center'].values()))
            dx = eye[0] - center[0]
            dy = eye[1] - center[1]
            x_ratio = dx / 100 + 0.5
            y_ratio = dy / 100 + 0.5
            x_ratio = max(0, min(1, x_ratio))
            y_ratio = max(0, min(1, y_ratio))
            return jsonify(x=x_ratio, y=y_ratio)
        except Exception as e:
            print(f"[ERROR] gaze mapping failed: {e}")
            return jsonify(x=-1, y=-1)

@app.route('/calibrate', methods=['POST'])
def calibrate():
    with lock:
        if eye_position:
            calibration_data['center'] = eye_position
            return jsonify(status='success')
        else:
            return jsonify(status='failed')

if __name__ == '__main__':
    app.run(debug=True)
