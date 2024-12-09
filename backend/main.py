import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pre-trained FaceNet model for feature extraction
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(img):
    img = img.convert("RGB")  # Ensure image is in RGB format
    img_resized = img.resize((160, 160))
    img_tensor = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    with torch.no_grad():
        img_embedding = facenet(img_tensor)
    return img_embedding.cpu().numpy().flatten()

# Load and preprocess the single input image
input_image_path = "img.png"  # Change this to the path of your input image
input_img = Image.open(input_image_path)
input_face_encoding = preprocess_image(input_img)

def match_face(face_encoding, input_face_encoding, threshold=0.6):
    similarity = cosine_similarity([face_encoding], [input_face_encoding])[0][0]
    return similarity, similarity > threshold

def process_frame(frame, frame_count):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)).resize((160, 160))
            face_encoding = preprocess_image(face_pil)
            similarity, is_match = match_face(face_encoding, input_face_encoding)
            print(f"Frame {frame_count}: Cosine Similarity = {similarity}")
            if is_match:
                print(f"Match found in frame {frame_count} (Similarity: {similarity:.2f})")
                return frame
    return None

def process_skipped_frames(skipped_frames):
    # Only check the previous frame if the current frame has a match
    if skipped_frames:
        frame, frame_count = skipped_frames[-1]
        processed_frame = process_frame(frame, frame_count)
        if processed_frame is not None:
            out.write(processed_frame)

input_video_path = "Paurashpur.mkv"  # Change this to the path of your input video
output_video_path = "output.mp4"

cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Set output FPS to match input video

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
batch_size = 32
frames = []
skip_frames = 0  # Initialize skip_frames counter
skipped_frames = []  # To store skipped frames
min_skip_frames = 2  # Minimum number of frames to skip
max_skip_frames = 3  # Maximum number of frames to skip
adaptive_skip = min_skip_frames  # Start with minimum skipping

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if skip_frames > 0:
        skipped_frames.append((frame, frame_count))
        skip_frames -= 1
        continue

    frames.append((frame, frame_count))

    if len(frames) == batch_size:
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results = list(executor.map(lambda f: process_frame(f[0], f[1]), frames))
            faces_detected = False
            for result in results:
                if result is not None:
                    out.write(result)
                    faces_detected = True
                    process_skipped_frames(skipped_frames)
                    skipped_frames.clear()
            if not faces_detected:
                adaptive_skip = min(adaptive_skip + 1, max_skip_frames)
                skip_frames = adaptive_skip  # Set to skip the next frames adaptively
            else:
                adaptive_skip = min_skip_frames  # Reset to minimum skipping if faces are found
        frames = []

# Process remaining frames
if frames:
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(lambda f: process_frame(f[0], f[1]), frames))
        for result in results:
            if result is not None:
                out.write(result)

cap.release()
out.release()
print(f"Trimmed video saved as {output_video_path}")
