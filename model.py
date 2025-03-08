import timm
import torch
import torch.nn as nn
from torchvision import transforms

def get_model(model_name="efficientnet_b3", num_classes=2, pretrained=True):
    """Khởi tạo mô hình phân loại deepfake với các mô hình mạnh hơn."""
    
    model = timm.create_model(model_name, pretrained=pretrained)
    in_features = model.get_classifier().in_features  # Lấy số feature đầu ra
    model.reset_classifier(num_classes)  # Reset classifier đúng cách
    return model


    return model

def load_model(model, model_path):
    """Load trọng số đã huấn luyện."""
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    return model
