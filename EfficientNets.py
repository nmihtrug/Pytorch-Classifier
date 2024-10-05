import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def conv_bn(inp, oup, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

def conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def SqueezeExcitation(in_channels, reduction=16):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
        nn.Sigmoid()
    )

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.out_channels = out_channels

        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU(inplace=True),
        ) if expand_ratio != 1 else nn.Identity()

        self.dw_conv = conv_dw(in_channels * expand_ratio, out_channels, stride)
        self.se = SqueezeExcitation(out_channels)

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x

        x = self.expand_conv(x)
        x = self.dw_conv(x)
        x = self.se(x)

        if self.use_res_connect:
            x += identity
        return x

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(EfficientNet, self).__init__()
        self.num_classes = num_classes

        # Define the configuration for EfficientNet
        self.configs = [
            # expand_ratio, channels, num_blocks, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 40, 2, 2],
            [6, 80, 3, 2],
            [6, 112, 3, 1],
            [6, 192, 4, 2],
            [6, 320, 1, 1]
        ]

        # Initial Conv Layer
        self.stem_conv = conv_bn(3, self.round_filters(32, width_mult), kernel_size=3, stride=2, padding=1)

        # Building the model
        layers = []
        in_channels = self.round_filters(32, width_mult)
        for expand_ratio, out_channels, num_blocks, stride in self.configs:
            for i in range(num_blocks):
                layers.append(MBConvBlock(in_channels, self.round_filters(out_channels, width_mult), expand_ratio, stride if i == 0 else 1))
                in_channels = self.round_filters(out_channels, width_mult)

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def round_filters(self, filters, width_mult):
        return int(filters * width_mult)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def efficientnet(version='B0', num_classes=1000):
    if version == 'B0':
        return EfficientNet(num_classes=num_classes, width_mult=1.0)
    elif version == 'B1':
        return EfficientNet(num_classes=num_classes, width_mult=1.0)  # Adjust widths based on B1 specs
    elif version == 'B2':
        return EfficientNet(num_classes=num_classes, width_mult=1.1)  # Adjust widths based on B2 specs
    elif version == 'B3':
        return EfficientNet(num_classes=num_classes, width_mult=1.2)  # Adjust widths based on B3 specs
    elif version == 'B4':
        return EfficientNet(num_classes=num_classes, width_mult=1.4)  # Adjust widths based on B4 specs
    elif version == 'B5':
        return EfficientNet(num_classes=num_classes, width_mult=1.6)  # Adjust widths based on B5 specs
    elif version == 'B6':
        return EfficientNet(num_classes=num_classes, width_mult=1.8)  # Adjust widths based on B6 specs
    elif version == 'B7':
        return EfficientNet(num_classes=num_classes, width_mult=2.0)  # Adjust widths based on B7 specs
    else:
        raise ValueError("Unsupported EfficientNet version")

if __name__ == "__main__":
    # Example usage
    model = efficientnet('B7', num_classes=1000)
    summary(model, (3, 224, 224), device='cpu')
