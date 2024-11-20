from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from PIL import Image
import os
import cv2

app = Flask(__name__)

# Load YOLOv5 model
MODEL_PATH = r'C:\Users\Admin\Desktop\indutrial_pro\flask-app-object-detecction\best.pt'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# Define upload and processed folders
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('file')
    if not file or file.filename == '':
        return "No file uploaded or invalid file", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    if file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        processed_file_path = process_image(file_path)
        file_type = 'image'
    elif file.filename.lower().endswith(('mp4', 'avi')):
        processed_file_path = process_video(file_path)
        file_type = 'video'
    else:
        return "Unsupported file format", 400

    uploaded_url = url_for('static', filename=f'uploads/{os.path.basename(file_path)}')
    processed_url = url_for('static', filename=f'processed/{os.path.basename(processed_file_path)}')
    return render_template('index.html', uploaded_file=uploaded_url, processed_file=processed_url, file_type=file_type)

def process_image(file_path):
    results = model(file_path)
    processed_path = os.path.join(PROCESSED_FOLDER, f'processed_{os.path.basename(file_path)}')
    results[0].save(processed_path)
    return processed_path

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    output_path = os.path.join(PROCESSED_FOLDER, f'processed_{os.path.basename(file_path)}')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        out.write(results[0].plot())

    cap.release()
    out.release()
    return output_path

if __name__ == '__main__':
    app.run(debug=True)
