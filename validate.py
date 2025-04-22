import os
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt  

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import ResNet50WithMask
from dataset import KidneyDataset, get_transform


def load_data(csv_file, base_path):
    df = pd.read_csv(csv_file)
    df['mask_path'] = df['file_paths'].apply(lambda x: os.path.join(base_path, "mask", *x.split("/")[-3:]))
    df = df[df['mask_path'].apply(os.path.exists)].reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.25, random_state=42)

    test_data_dicts = []
    for index, row in test_df.iterrows():
        image = row["full_path"]
        mask = row["mask_path"]
        label = 0 if row['kidney_healthy'] == 1 else 1
        test_data_dicts.append({'image': os.path.basename(image), 'image_path': image, 'mask_path': mask, 'label': label})

    return test_data_dicts

def evaluate_model(model_path, csv_file, base_path, batch_size=8, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data_dicts = load_data(csv_file, base_path)
    transform = get_transform()
    test_dataset = KidneyDataset(test_data_dicts, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ResNet50WithMask(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(images, masks)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

def evaluate_model_after_train(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(images, masks)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # 混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["kidney_health", "kidney_low", "kidney_high"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    png_out_path = "confusion_matrix.png"
    plt.savefig(png_out_path)

    return cm

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate kidney injury classification model")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to the saved model checkpoint")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file with data info")
    parser.add_argument("--base_path", type=str, required=True, help="Base path for nifti files")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for validation")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.csv_file, args.base_path, args.batch_size)
