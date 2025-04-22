import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
import random
import numpy as np

def get_augmentation_for_imbalanced_class():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),                # 隨機水平翻轉
        transforms.RandomRotation(degrees=30),                  # 隨機旋轉 ±30 度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 顏色抖動
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 隨機仿射變換
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),    # 隨機裁剪 + 調整大小
        transforms.RandomVerticalFlip(p=0.5),                   # 隨機垂直翻轉
        transforms.RandomRotation(degrees=90),                   # 隨機90度旋轉
        transforms.ToTensor(),                                  # 轉為 tensor
        transforms.Normalize(mean=[0.5], std=[0.5])              # 標準化
    ])

def get_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),                # 隨機水平翻轉
        transforms.RandomRotation(degrees=15),                 # 隨機旋轉 ±15 度
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 顏色抖動
        transforms.ToTensor(),                                 # 轉為 tensor
        transforms.Normalize(mean=[0.5], std=[0.5])            # 標準化
    ])

def add_noise_to_image(image, noise_factor=0.1):
    """
    加入隨機噪聲（高斯噪聲）到影像。
    """
    noise = torch.randn_like(image) * noise_factor
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0., 1.)

def apply_transform_based_on_label(image, label):
    """
    根據樣本的標籤決定使用哪一種增強。
    """
    if label == 1:  # Kidney Low
        # 對 kidney_low 樣本使用強增強
        transform = get_augmentation_for_imbalanced_class()
        image = transform(image)
    elif label == 2:  # Kidney High
        # 對 kidney_high 樣本使用強增強
        transform = get_augmentation_for_imbalanced_class()
        image = transform(image)
    else:
        # 對健康樣本使用一般增強
        transform = get_transform()
        image = transform(image)

    return image

class KidneyDataset(Dataset):
    def __init__(self, data_dicts, transform=None, target_size=(512, 512)):
        self.data_dicts = data_dicts
        self.transform = transform
        self.target_size = target_size  # 統一 resize 尺寸

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        sample = self.data_dicts[idx]
        image_path = sample['image_path']
        mask_path = sample['mask_path']
        label = sample['label']

        # 讀取 NIfTI 檔案
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        image_data = image_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        image_data = image_data.squeeze()
        mask_data = mask_data.squeeze()

        # 取中央切片
        if image_data.ndim == 3:
            image_slice = image_data[:, :, image_data.shape[2] // 2]
        elif image_data.ndim == 2:
            image_slice = image_data
        else:
            raise ValueError(f"Unsupported image dimension: {image_data.ndim}")

        if mask_data.ndim == 3:
            mask_slice = mask_data[:, :, mask_data.shape[2] // 2]
        elif mask_data.ndim == 2:
            mask_slice = mask_data
        else:
            raise ValueError(f"Unsupported mask dimension: {mask_data.ndim}")

        # 轉 tensor + 增加 channel 維度 (C, H, W)
        image_tensor = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0)

        # 將灰階影像轉換為三通道影像
        image_tensor = image_tensor.repeat(3, 1, 1)

        # ➕ resize 到固定尺寸 (1, H, W) → (1, 512, 512)
        image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        image_pil = to_pil_image(image_tensor)
        
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)

        # ➕ Normalization + 增強
        image_tensor = apply_transform_based_on_label(image_pil, label)

        return image_tensor, mask_tensor, label
