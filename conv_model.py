import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Level 1
        self.conv1_1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        # Level 2
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.drop2_1 = nn.Dropout(0.1)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Level 3
        self.pool3_1 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.up3_1 = nn.Upsample(scale_factor=2, mode='nearest')

        # Level 2 again
        self.conv2_3 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.drop2_2 = nn.Dropout(0.1)
        self.conv2_4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.up2_1 = nn.Upsample(scale_factor=2, mode='nearest')

        # Level 1 again
        self.conv1_3 = nn.Conv2d(48, 16, kernel_size=3, padding=1)
        self.conv1_4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.output = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Level 1
        x1 = F.relu(self.conv1_1(x))
        x1 = F.relu(self.conv1_2(x1))

        # Level 2
        x2 = self.pool2_1(x1)
        x2 = F.relu(self.conv2_1(x2))
        x2 = self.drop2_1(x2)
        x2 = F.relu(self.conv2_2(x2))

        # Level 3
        x3 = self.pool3_1(x2)
        x3 = F.relu(self.conv3_1(x3))
        x3 = F.relu(self.conv3_2(x3))
        x3 = F.relu(self.conv3_3(x3))
        x3 = F.relu(self.conv3_4(x3))
        x3 = self.up3_1(x3)

        # Level 2 again
        x2 = torch.cat([x2, x3], dim=1)
        x2 = F.relu(self.conv2_3(x2))
        x2 = self.drop2_2(x2)
        x2 = F.relu(self.conv2_4(x2))
        x2 = self.up2_1(x2)

        # Level 1 again
        x1 = torch.cat([x1, x2], dim=1)
        x1 = F.relu(self.conv1_3(x1))
        x1 = F.relu(self.conv1_4(x1))
        output = torch.sigmoid(self.output(x1))
        
        return output

def build():
    return CustomModel()
