from flask import Flask, request, jsonify, send_from_directory, abort
from app import trim_video  # Ensure trim_video is correctly imported
import os
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_files():
    video = request.files['video']
    face = request.files['face']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    face_path = os.path.join(UPLOAD_FOLDER, face.filename)
    output_path = os.path.join(UPLOAD_FOLDER, f'trimmed_{video.filename}')
    
    video.save(video_path)
    face.save(face_path)
    
    try:
        with Image.open(face_path) as img:
            img.verify()
    except (IOError, SyntaxError) as e:
        app.logger.error(f"Invalid image file: {face_path}, error: {e}")
        return jsonify({"error": "Invalid image file"}), 400

    trim_video(video_path, face_path, output_path)
    
    trimmed_video_url = f'http://http://127.0.0.1:5000/uploads/{os.path.basename(output_path)}'
    return jsonify({"trimmedVideoUrl": trimmed_video_url})

@app.route('/uploads/<path:filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except FileNotFoundError:
        app.logger.error(f"File not found: {filename}")
        abort(404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
