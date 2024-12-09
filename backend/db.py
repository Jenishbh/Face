from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from app import trim_video  # Ensure trim_video is correctly imported
import os
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'png', 'jpg', 'jpeg'}

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to validate file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'video' not in request.files or 'face' not in request.files:
        return jsonify({"error": "Missing video or face file"}), 400

    video = request.files['video']
    face = request.files['face']

    if not allowed_file(video.filename) or not allowed_file(face.filename):
        return jsonify({"error": "Invalid file format"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    face_path = os.path.join(UPLOAD_FOLDER, face.filename)
    output_path = os.path.join(UPLOAD_FOLDER, f'trimmed_{video.filename}')

    video.save(video_path)
    face.save(face_path)

    # Validate the uploaded image
    try:
        with Image.open(face_path) as img:
            img.verify()
    except (IOError, SyntaxError) as e:
        app.logger.error(f"Invalid image file: {face_path}, error: {e}")
        return jsonify({"error": "Invalid image file"}), 400

    # Process the video
    try:
        trim_video(video_path, face_path, output_path)
    except Exception as e:
        app.logger.error(f"Error in video processing: {e}")
        return jsonify({"error": "Video processing failed"}), 500

    trimmed_video_url = f'{request.host_url}uploads/{os.path.basename(output_path)}'
    app.logger.info(f"Trimmed video available at: {trimmed_video_url}")
    return jsonify({"trimmedVideoUrl": trimmed_video_url})

@app.route('/uploads/<path:filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except FileNotFoundError:
        app.logger.error(f"File not found: {filename}")
        abort(404)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Allow dynamic port configuration
    app.run(host='0.0.0.0', port=port)
