import os
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
import argparse


def extract_faces(video_path, output_folder, confidence_threshold=0.9):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)

    frame_list = range(0, total_frames, fps)  
    frames_data = []
    frame_index = 0
    for frame_id in tqdm(frame_list, desc="Processing video"): 
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, probs = mtcnn.detect(frame_rgb)

        if faces is None or probs is None:
            continue

        for j, (face, prob) in enumerate(zip(faces, probs)):
            if prob is None or prob < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, face)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x1 >= x2 or y1 >= y2:
                continue

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            frames_data.append((frame_id, (x1, y1, x2, y2), face_crop))

            output_path = os.path.join(output_folder, f"frame_{frame_index}_face_{j}.jpg")
            cv2.imwrite(output_path, face_crop)
        frame_index += 1 
    
    cap.release()
    return frames_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from a video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save extracted faces")
    parser.add_argument("--confidence_threshold", type=float, default=0.9, help="Confidence threshold for face detection")

    args = parser.parse_args()

    extract_faces(args.video_path, args.output_folder, args.confidence_threshold)