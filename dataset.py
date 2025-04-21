import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
from torchvision import transforms

class KidneyDataset(Dataset):
    def __init__(self, data_dicts, transform=None):
        self.data_dicts = data_dicts
        self.transform = transform

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        sample = self.data_dicts[idx]
        image_path = sample['image_path']
        mask_path = sample['mask_path']
        label = sample['label']

        # 讀取 nifti 影像和 mask
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        # 取出影像資料
        image_data = image_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        # 去除單維度 (squeeze) 以處理像 (1024, 1, 1024, 1) 這樣的情況
        image_data = image_data.squeeze()
        mask_data = mask_data.squeeze()

        # 判斷影像維度，處理非3D影像
        if image_data.ndim == 3:
            mid_slice = image_data.shape[2] // 2
            image_slice = image_data[:, :, mid_slice]
        elif image_data.ndim == 2:
            image_slice = image_data
        else:
            raise ValueError(f"Unsupported image dimension: {image_data.ndim}")

        if mask_data.ndim == 3:
            mid_slice = mask_data.shape[2] // 2
            mask_slice = mask_data[:, :, mid_slice]
        elif mask_data.ndim == 2:
            mask_slice = mask_data
        else:
            raise ValueError(f"Unsupported mask dimension: {mask_data.ndim}")

        # 將 numpy array 轉成 tensor，並加上 channel 維度
        image_tensor = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0)

        # 正規化影像
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor, label

def get_transform():
    return transforms.Normalize(mean=[0.5], std=[0.5])
