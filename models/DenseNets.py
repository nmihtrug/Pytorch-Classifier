import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate=0.2):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.dropout(out)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out, inplace=True)
        out = self.conv(out)
        out = self.dropout(out)
        out = F.avg_pool2d(out, kernel_size=2, stride=2)
        return out

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, dropout_rate=0.2):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate, dropout_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat((x, new_features), 1)
        return x

class DenseNet(nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32, block_config=(6, 12, 24, 16), dropout_rate=0.2):
        super(DenseNet, self).__init__()
        num_init_features = 2 * growth_rate

        # Initial convolution
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks
        self.dense1 = DenseBlock(num_init_features, block_config[0], growth_rate, dropout_rate)
        num_features = num_init_features + block_config[0] * growth_rate
        self.trans1 = Transition(num_features, num_features // 2, dropout_rate)

        self.dense2 = DenseBlock(num_features // 2, block_config[1], growth_rate, dropout_rate)
        num_features = num_features // 2 + block_config[1] * growth_rate
        self.trans2 = Transition(num_features, num_features // 2, dropout_rate)

        self.dense3 = DenseBlock(num_features // 2, block_config[2], growth_rate, dropout_rate)
        num_features = num_features // 2 + block_config[2] * growth_rate
        self.trans3 = Transition(num_features, num_features // 2, dropout_rate)

        self.dense4 = DenseBlock(num_features // 2, block_config[3], growth_rate, dropout_rate)
        num_features = num_features // 2 + block_config[3] * growth_rate

        # Final batch norm and linear layer
        self.bn2 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Initial Conv Layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Dense Blocks and Transitions
        x = self.dense1(x)
        x = self.trans1(x)

        x = self.dense2(x)
        x = self.trans2(x)

        x = self.dense3(x)
        x = self.trans3(x)

        x = self.dense4(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Global Average Pooling
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def densenet(version='DenseNet121', num_classes=1000):
    if version == 'DenseNet121':
        return DenseNet(num_classes=num_classes, block_config=(6, 12, 24, 16))
    elif version == 'DenseNet169':
        return DenseNet(num_classes=num_classes, block_config=(6, 12, 32, 32))
    elif version == 'DenseNet201':
        return DenseNet(num_classes=num_classes, block_config=(6, 12, 48, 32))
    elif version == 'DenseNet264':
        return DenseNet(num_classes=num_classes, block_config=(6, 12, 64, 48))
    else:
        raise ValueError("Unsupported DenseNet version")

if __name__ == "__main__":
    # Example usage
    model = densenet('DenseNet264', num_classes=1000)
    summary(model, (3, 224, 224), device='cpu')
