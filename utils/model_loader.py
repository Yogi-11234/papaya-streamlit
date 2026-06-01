import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch,
                 kernel=3, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_ch,
                out_ch,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
        )

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    1,
                    bias=False
                ),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out = self.relu(out + residual)

        return self.pool(out)


class PapayaLeafCNN(nn.Module):

    def __init__(self,
                 num_classes=5,
                 dropout=0.3):
        super().__init__()

        self.block1 = nn.Sequential(
            ConvBNReLU(3, 32),
            nn.MaxPool2d(2, 2),
        )

        self.block2 = nn.Sequential(
            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(2, 2),
        )

        self.block3 = ResConvBlock(64, 128)
        self.block4 = ResConvBlock(128, 256)
        self.block5 = ResConvBlock(256, 512)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.gap(x)

        return self.classifier(x)


def load_model(model_path):

    model = PapayaLeafCNN(
        num_classes=5,
        dropout=0.3
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model