import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from detect_from_video import extract_faces
from torchvision import transforms

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_frames(frames_folder, model, device):
    """Dự đoán từng frame trong thư mục"""
    
    # Đưa model về chế độ đánh giá
    model.eval()
    
    # Dự đoán trên từng frame
    predictions = []
    with torch.no_grad():
        for frame in os.listdir(frames_folder):
            img_path = os.path.join(frames_folder, frame)
            img = Image.open(img_path).convert("RGB")
            img = transformer(img).unsqueeze(0).to(device)
            output = model(img)

            # Lấy xác suất của lớp "REAL" (class 1)
            prob_real = F.softmax(output, dim=1)[:, 1].item()
            predictions.append(prob_real)

    # Tính toán kết quả cuối cùng
    avg_prob_real = np.mean(predictions)
    label = 1 if avg_prob_real >= 0.5 else 0


    return label, avg_prob_real


def predict_video(video_folder, model, device):
    """Dự đoán toàn bộ video"""
    # Đưa model về chế độ đánh giá
    model.eval()
    
    # Dự đoán trên từng frame
    predictions = []
    frames = extract_faces(video_folder)
    with torch.no_grad():
        for frame in frames:
            output = model(frame.to(device))

            # Lấy xác suất của lớp "REAL" (class 1)
            prob_real = F.softmax(output, dim=1)[:, 1].item()
            predictions.append(prob_real)

    # Tính toán kết quả cuối cùng
    avg_prob_real = np.mean(predictions)
    label = "REAL" if avg_prob_real >= 0.5 else "FAKE"


    return label, avg_prob_real

    

import argparse
from model import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict deepfake detection on video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference")

    args = parser.parse_args()

    # Load model
    model = get_model()
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)

    # Extract faces & predict
    extracted_faces_folder = "temp_faces"
    extract_faces(args.video_path, extracted_faces_folder)
    label, confidence = predict_frames(extracted_faces_folder, model, args.device)

    print(f"Predicted: {'REAL' if label == 1 else 'FAKE'} with confidence: {confidence:.4f}")
