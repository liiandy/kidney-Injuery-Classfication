import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from collections import Counter
# from dataset import KidneyDataset, get_transform
from dataset_new import KidneyDataset
from train import train_model
from validate import evaluate_model, evaluate_model_after_train

# 🔻 1. 讀取 CSV 檔案
csv_file = "/workspace/data/a313112015/rsna_train_new_v2.csv"
df = pd.read_csv(csv_file)


base_path = "/workspace/data/a313112015/BiomedParse/out_test/nifti/"

# 新增一欄 'mask_path' 到 df
df['mask_path'] = df['file_paths'].apply(lambda x: os.path.join(base_path, "kidney", *x.split("/")[-3:]))


df = df[df['mask_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"✅ Total samples with mask found: {len(df)}")

# 🔻 6:3:1 分割
train_df, temp_df = train_test_split(
    df,
    test_size=0.4,  # 驗證 + 測試
    random_state=42,
    stratify=df["any_injury"]    # ← 根據類別標籤進行平衡分割
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.25,  # 10% 最後給測試
    random_state=42,
    stratify=temp_df["any_injury"]    # ← 根據類別標籤進行平衡分割
)

# 印出每個 set 的資料數量
print(f"Train: {len(train_df)}")
print(train_df["any_injury"].value_counts())
print("\n")

print(f"Valid: {len(valid_df)}")
print(valid_df["any_injury"].value_counts())
print("\n")

print(f"Test: {len(test_df)}")
print(test_df["any_injury"].value_counts())
print("\n")


# +
def data_progress(df, dicts):
    for index, row in df.iterrows():
        image = row["file_paths"]
        # image_path = image
        # gt = f'/tf/angela0503/Spleen_data/RAS_dilation_10/{row["spleen_injury"]}_{row["chartNo"]}@venous.nii.gz' 
        mask = row["mask_path"]
        
        
        if row['any_injury'] == 1:
            label = 1
        else:
            label = 0
        
#         if row['kidney_low'] == 1:
#             label = 1
#         elif row['kidney_high'] == 1:
#             label = 2  
#         else:
#             label = 0

        dicts.append({'image':os.path.basename(image), 'image_path':image, 'mask_path':mask, 'label':label})
    return dicts
# -

train_data_dicts = []
valid_data_dicts = []
test_data_dicts = []
train_data_dicts = data_progress(train_df,train_data_dicts)
valid_data_dicts = data_progress(valid_df,valid_data_dicts)
test_data_dicts = data_progress(test_df,test_data_dicts)

print(f'\n Train:{len(train_data_dicts)},Valid:{len(valid_data_dicts)},Test:{len(test_data_dicts)}')


labels = [item['label'] for item in train_data_dicts]
print(Counter(labels))

# +
# transform = get_transform()
# -

# train_dataset = KidneyDataset(train_data_dicts, transform=transform)
# valid_dataset = KidneyDataset(valid_data_dicts, transform=transform)
# test_dataset = KidneyDataset(test_data_dicts, transform=transform)
train_dataset = KidneyDataset(train_data_dicts)
valid_dataset = KidneyDataset(valid_data_dicts)
test_dataset = KidneyDataset(test_data_dicts)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

trained_model = train_model(train_loader, valid_loader, mode='NoMask', num_classes=len(set(labels)), num_epochs=1)

# Use validate.py's evaluate_model function for evaluation with test_loader
cm = evaluate_model_after_train(trained_model, test_loader)
