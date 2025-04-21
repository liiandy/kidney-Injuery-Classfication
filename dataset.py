import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F
from torchvision import transforms

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
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)

        # ➕ Normalization
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor, label



def get_transform():
    return transforms.Normalize(mean=[0.5], std=[0.5])
