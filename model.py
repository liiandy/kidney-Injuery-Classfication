import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50WithMask(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50WithMask, self).__init__()
        # 載入預設範本 ResNet50 模型（不包含最後的全連接層）
        self.resnet50 = models.resnet50(pretrained=True)
        self.conv1 = self.resnet50.conv1
        self.bn1 = self.resnet50.bn1
        self.relu = self.resnet50.relu
        self.maxpool = self.resnet50.maxpool
        self.layer1 = self.resnet50.layer1
        self.layer2 = self.resnet50.layer2
        self.layer3 = self.resnet50.layer3
        self.layer4 = self.resnet50.layer4
        self.avgpool = self.resnet50.avgpool
        
        # 側支：處理 mask
        self.mask_branch = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 合併層：將主支和側支合併（使用加法）
        self.merge = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 分類頭
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, image, mask):
        # 主支：處理影像
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 側支：處理 mask
        mask_features = self.mask_branch(mask)
        
        # 合併：將主支和側支特征相加
        combined = x + mask_features
        combined = self.merge(combined)
        
        # 繼續通過 ResNet50 的其餘層
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
    
# 示例：初始化模型
model = ResNet50WithMask(num_classes=10)  # 假設有 10 個類別

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 訓練循環（示例）
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for images, masks, labels in train_loader:
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
