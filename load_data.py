from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Chuẩn bị transform cho ảnh
transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class LoadData(Dataset):
    def __init__(self, root_dir, transform=transformer):
        self.root_dir = root_dir
        self.transform = transform
        self.frames = []  # Lưu tất cả các frame và nhãn

        # Duyệt qua thư mục FAKE & REAL
        for label in ['fake', 'real']:
            label_dir = os.path.join(root_dir, label)
            if not os.path.exists(label_dir):
                continue  # Bỏ qua nếu thư mục không tồn tại
            
            # Duyệt qua từng thư mục video
            for video_dir in os.listdir(label_dir):
                video_path = os.path.join(label_dir, video_dir)
                if os.path.isdir(video_path):
                    # Lấy danh sách ảnh hợp lệ trong video
                    frame_paths = sorted([
                        os.path.join(video_path, f) for f in os.listdir(video_path)
                        if f.lower().endswith(('.jpg', '.png'))
                    ])

                    # Lưu tất cả frame và nhãn
                    self.frames.extend([(frame, 0 if label == 'fake' else 1) for frame in frame_paths])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path, label = self.frames[idx]

        try:
            image = Image.open(frame_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {frame_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label
