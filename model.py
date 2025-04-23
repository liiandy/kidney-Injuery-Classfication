import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet50WithMask(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50WithMask, self).__init__()
        # 載入預訓練的 ResNet50（使用新 weights 寫法）
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 主分支 - 影像處理
        self.conv1 = self.resnet50.conv1
        self.bn1 = self.resnet50.bn1
        self.relu = self.resnet50.relu
        self.maxpool = self.resnet50.maxpool
        self.layer1 = self.resnet50.layer1
        self.layer2 = self.resnet50.layer2
        self.layer3 = self.resnet50.layer3
        self.layer4 = self.resnet50.layer4
        self.avgpool = self.resnet50.avgpool

        # 側分支 - mask 處理
        self.mask_branch = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 合併主支與側支
        self.merge = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 分類器
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, image, mask):
        # 主支：處理影像
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)

        # 側支：處理 mask
        mask_features = self.mask_branch(mask)

        # ⭐️ Resize mask 特徵圖，讓尺寸與 x 一致
        if mask_features.shape[2:] != x.shape[2:]:
            mask_features = F.interpolate(mask_features, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 合併兩個分支的特徵
        combined = x + mask_features
        combined = self.merge(combined)

        # ResNet 下游結構
        x = self.maxpool(combined)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 分類
        out = self.classifier(x)
        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # 載入預訓練的 ResNet50（使用新 weights 寫法）
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 主分支 - 影像處理
        self.conv1 = self.resnet50.conv1
        self.bn1 = self.resnet50.bn1
        self.relu = self.resnet50.relu
        self.maxpool = self.resnet50.maxpool
        self.layer1 = self.resnet50.layer1
        self.layer2 = self.resnet50.layer2
        self.layer3 = self.resnet50.layer3
        self.layer4 = self.resnet50.layer4
        self.avgpool = self.resnet50.avgpool

        # 分類器
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, image, mask):
        # 主支：處理影像
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)

        # ResNet 下游結構
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 分類
        out = self.classifier(x)
        return out
