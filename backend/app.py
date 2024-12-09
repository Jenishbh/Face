import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import torchreid
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load models
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
pose_model = YOLO('yolov8n-pose.pt')
reid_model = torchreid.models.build_model(
    name='osnet_x1_0', num_classes=1000, pretrained=True
).eval().to(device)

# Transforms
face_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

body_transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def normalize_embeddings(embeddings):
    return embeddings / embeddings.norm(dim=-1, keepdim=True)

def preprocess_reference_image(reference_image_path):
    """Preprocess and extract features from a single reference image."""
    img = cv2.imread(reference_image_path)
    faces, confidences = detect_faces(img)
    if not faces:
        raise ValueError("No face detected in reference image!")
    
    best_face = faces[np.argmax(confidences)]
    face_crop = img[best_face[1]:best_face[3], best_face[0]:best_face[2]]
    face_features = extract_face_features(face_crop)
    body_crop = extract_body_area(img, best_face)
    body_features = extract_body_features(body_crop)
    clothing_features = extract_clothing_features(body_crop)
    return {
        'face_features': face_features,
        'body_features': body_features,
        'clothing_features': clothing_features
    }

def detect_faces(image):
    """Detect faces in an image using Haar Cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces, [1.0] * len(faces)  # Dummy confidence for Haar Cascade

def extract_face_features(face_img):
    """Extract facial embeddings using FaceNet."""
    face_tensor = face_transform(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    with torch.no_grad():
        features = facenet(face_tensor)
    return normalize_embeddings(features).cpu().numpy()

def extract_body_features(body_img):
    """Extract body features using ReID model."""
    body_tensor = body_transform(Image.fromarray(cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    with torch.no_grad():
        features = reid_model(body_tensor)
    return features.cpu().numpy()

def extract_clothing_features(img):
    """Extract clothing features such as color, texture, and pattern."""
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist([img_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(color_hist, color_hist)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    pattern_density = np.mean(edges > 0)

    radius = 3
    n_points = 8
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)

    return {
        'color_hist': color_hist,
        'pattern_density': pattern_density,
        'texture_hist': lbp_hist
    }

def extract_body_area(frame, face):
    """Extract the body region based on the face location."""
    x1, y1, x2, y2 = face
    body_y1 = max(0, y1 - (y2 - y1))
    body_y2 = min(frame.shape[0], y2 + (y2 - y1) * 2)
    body_x1 = max(0, x1 - (x2 - x1))
    body_x2 = min(frame.shape[1], x2 + (x2 - x1))
    return frame[body_y1:body_y2, body_x1:body_x2]

def match_features(input_features, reference_features, threshold=0.6):
    """Compute similarity between input and reference features."""
    face_sim = cosine_similarity([input_features['face_features']], [reference_features['face_features']])[0][0]
    body_sim = cosine_similarity([input_features['body_features']], [reference_features['body_features']])[0][0]
    cloth_sim = cosine_similarity([input_features['clothing_features']['color_hist'].ravel()],
                                   [reference_features['clothing_features']['color_hist'].ravel()])[0][0]
    final_similarity = 0.5 * face_sim + 0.3 * body_sim + 0.2 * cloth_sim
    return final_similarity > threshold, final_similarity

def collect_target_images(video_path, reference_features, output_folder, max_images=10):
    """Collect unique images of the target person from the video."""
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    unique_images = []
    print("Collecting target person images...")

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while len(unique_images) < max_images and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            pbar.update(1)

            faces, _ = detect_faces(frame)
            for face in faces:
                body_img = extract_body_area(frame, face)
                input_features = {
                    'face_features': extract_face_features(frame[face[1]:face[3], face[0]:face[2]]),
                    'body_features': extract_body_features(body_img),
                    'clothing_features': extract_clothing_features(body_img)
                }
                is_match, _ = match_features(input_features, reference_features)
                if is_match:
                    unique_images.append(body_img)
                    cv2.imwrite(os.path.join(output_folder, f"target_{len(unique_images)}.jpg"), body_img)
                    if len(unique_images) >= max_images:
                        break

    cap.release()
    print(f"Collected {len(unique_images)} images of the target person.")
    return unique_images

def trim_video(input_video_path, reference_features, output_video_path):
    """Trim video based on matched features."""
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    adaptive_skip = 1
    frame_number = 0
    print("Processing video...")
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            pbar.update(1)

            if frame_number % adaptive_skip != 0:
                continue

            faces, _ = detect_faces(frame)
            for face in faces:
                body_img = extract_body_area(frame, face)
                input_features = {
                    'face_features': extract_face_features(frame[face[1]:face[3], face[0]:face[2]]),
                    'body_features': extract_body_features(body_img),
                    'clothing_features': extract_clothing_features(body_img)
                }
                is_match, _ = match_features(input_features, reference_features)
                if is_match:
                    out.write(frame)
                    adaptive_skip = 1
                else:
                    adaptive_skip = min(adaptive_skip + 1, 5)

    cap.release()
    out.release()
    print(f"Trimmed video saved as {output_video_path}")

if __name__ == '__main__':
    input_video_path = "input_video.mp4"
    input_image_path = "input_face.png"
    output_folder = "target_images"
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Extract reference features from the provided image
    reference_features = preprocess_reference_image(input_image_path)

    # Step 2: Collect target person's unique images from the video
    target_images = collect_target_images(input_video_path, reference_features, output_folder, max_images=10)

    # Step 3: Use the collected images to trim the video
    output_video_path = "output_trimmed.mp4"
    trim_video(input_video_path, reference_features, output_video_path)
