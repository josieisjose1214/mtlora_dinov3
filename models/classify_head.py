import torch.nn as nn
 
class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化层
        self.flatten = nn.Flatten()  # 扁平化层
        self.fc = nn.Linear(dim, num_classes)  # 全连接层
 
    def forward(self, x):
        x = self.global_avg_pool(x[3])  # 应用全局平均池化
        x = self.flatten(x)  # 扁平化特征图
        x = self.fc(x)  # 应用全连接层得到类别预测
        return x