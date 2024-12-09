
# FaceTrim: Intelligent Face Detection and Video Trimming

FaceTrim is an advanced deep learning model designed for precise face detection and video trimming. This project leverages state-of-the-art technologies to detect, match, and trim video segments containing specific faces, providing an efficient solution for video processing and editing.

![FaceTrim](path_to_logo_or_demo_image.png)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Face Detection:** Utilizes Haar Cascade classifiers for initial face detection in video frames.
- **Feature Extraction:** Employs the InceptionResnetV1 model from `facenet_pytorch` for extracting facial features.
- **Face Matching:** Uses cosine similarity to match detected faces with a reference image.
- **Video Trimming and Merging:** Automatically trims video segments containing the matched face and merges them into a new video file.
- **GPU Acceleration:** Optimized for CUDA-compatible GPUs to enhance processing speed and efficiency.

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional but recommended for faster processing)

### Dependencies

Install the required Python packages using `pip`:

```bash
pip install torch torchvision facenet-pytorch opencv-python scikit-learn
```

## Usage

### Preprocess the Input Image

Prepare the reference image that you want to match in the video:

```python
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np

def preprocess_image(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((160, 160))
    img_tensor = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    with torch.no_grad():
        img_embedding = facenet(img_tensor)
    return img_embedding.cpu().numpy().flatten()

input_face_encoding = preprocess_image('path_to_reference_image.png')
```

### Process the Video

Run the face detection and video trimming:

```python
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

def preprocess_image_from_frame(face_image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)).resize((160, 160))
    img_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    with torch.no_grad():
        img_embedding = facenet(img_tensor)
    return img_embedding.cpu().numpy().flatten()

def match_face(face_encoding, input_face_encoding, threshold=0.6):
    similarity = cosine_similarity([face_encoding], [input_face_encoding])[0][0]
    return similarity, similarity > threshold

def process_video(input_video_path, output_video_path, input_face_encoding):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            face_encoding = preprocess_image_from_frame(face_image)

            if face_encoding is not None:
                similarity, is_match = match_face(face_encoding, input_face_encoding)
                if is_match:
                    out.write(frame)
                    break

    cap.release()
    out.release()
    print(f"Trimmed video saved as {output_video_path}")

process_video('path_to_input_video.mp4', 'output.mp4', input_face_encoding)
```

## Technologies Used

- **Programming Languages:** Python
- **Libraries and Frameworks:** PyTorch, OpenCV, facenet_pytorch, scikit-learn
- **Deep Learning Models:** InceptionResnetV1, Haar Cascade
- **Optimization:** CUDA for GPU acceleration




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


