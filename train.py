import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train(model, train_loader, val_loader, epochs, lr, device, save_path="best_model.pth"):
    """Huấn luyện mô hình trên toàn bộ video"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0  
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)  # Đưa toàn bộ frame vào model
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            

    print("raining complete! Best Val Acc:", best_acc)
    plot_training(train_losses, train_accuracies, val_losses, val_accuracies)

def plot_training(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label="Train Loss")
    plt.plot(epochs, val_losses, 'r--', label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'g-', label="Train Accuracy")
    plt.plot(epochs, val_accuracies, 'orange', label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    
    plt.show()

def evaluate(model, test_loader, device):
    """Đánh giá mô hình trên tập kiểm thử"""
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Accuracy: {acc:.2f} - Precision: {precision:.2f} - Recall: {recall:.2f} - F1: {f1:.2f}")

    confusion_matrix_result = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(confusion_matrix_result)
    

    return acc, precision, recall, f1

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--device", type=str, required=True, choices=["cpu", "cuda"], help="Device to train on")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Path to save the best model")

    args = parser.parse_args()

    # Load model và dữ liệu (cần đảm bảo bạn đã có model và dataloader)
    from model import get_model
    from load_data import LoadData
    from torch.utils.data import DataLoader

    # Khởi tạo mô hình
    model = get_model()

    # Load dữ liệu
    train_dataset = LoadData(root_dir="data/train")
    val_dataset = LoadData(root_dir="data/val")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Gọi hàm train
    train(model, train_loader, val_loader, args.epochs, args.lr, args.device, args.save_path)
