"""All the models, going from simplest to fanciest. Grab one with build_model(name).

  mlp         - just flatten + dense layers. knows nothing about space -> underfits.
  tiny_cnn    - 2 conv blocks. the first actual cnn.
  vgg         - vgg-ish stack where bn / dropout / depth are all config flags. I reuse the
                same class for the "no regularization -> overfits" run and the
                "add regularization -> generalizes" run, just with a different config.
  resnet_mini - small resnet (with skip connections), trained from scratch.
  resnet18    - torchvision's resnet18 hacked to take 1-channel input (transfer learning).

I use LazyLinear in the spots where the flattened size depends on the input size, so the
models don't break if img_size changes. The catch: you have to run one forward pass first
so those lazy layers actually get built (train.py does this before counting params).
"""
import torch
import torch.nn as nn

from .utils import NUM_CLASSES


# ===== 1. MLP baseline =====
class EmotionMLP(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden=(512, 256), dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]
        for h in hidden:
            layers += [nn.LazyLinear(h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.LazyLinear(num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===== 2. Tiny CNN =====
class TinyCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ===== 3. Configurable VGG-style net =====
def _conv_block(in_c, out_c, bn):
    # if we use batchnorm the conv doesn't need a bias (bn has its own)
    layers = [nn.Conv2d(in_c, out_c, 3, padding=1, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.ReLU(inplace=True))
    return layers


class VGGStyle(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, channels=(64, 128, 256),
                 bn=True, dropout=0.5, convs_per_block=2):
        super().__init__()
        feats = []
        in_c = 1
        for c in channels:
            for i in range(convs_per_block):
                feats += _conv_block(in_c if i == 0 else c, c, bn)
            feats.append(nn.MaxPool2d(2))
            if dropout > 0:
                feats.append(nn.Dropout2d(dropout * 0.5))
            in_c = c
        self.features = nn.Sequential(*feats)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # so the head doesn't care about the exact spatial size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ===== 4. Mini ResNet (from scratch) =====
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        # if the shape changes we need a 1x1 conv on the skip path to match it
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # the skip connection
        return torch.relu(out)


class ResNetMini(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, channels=(64, 128, 256),
                 blocks_per_stage=2, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True),
        )
        stages = []
        in_c = channels[0]
        for i, c in enumerate(channels):
            for b in range(blocks_per_stage):
                stride = 2 if (b == 0 and i > 0) else 1   # halve the resolution at the start of each new stage
                stages.append(BasicBlock(in_c, c, stride))
                in_c = c
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


# ===== 5. Transfer learning: torchvision ResNet18 made grayscale =====
def build_resnet18(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)

    # resnet's first conv expects 3 channels (rgb) but our faces are grayscale (1 channel).
    # so swap in a 1-channel conv. if it's pretrained, average the rgb filters into the new
    # one so we keep the learned edge/texture filters instead of starting from scratch.
    old_w = m.conv1.weight.data
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        m.conv1.weight.data = old_w.mean(dim=1, keepdim=True)

    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False

    m.fc = nn.Linear(m.fc.in_features, num_classes)  # brand new head for our 7 classes (always trains)
    return m


# ===== factory =====
def build_model(name: str, num_classes: int = NUM_CLASSES, **params) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return EmotionMLP(num_classes, **params)
    if name == "tiny_cnn":
        return TinyCNN(num_classes, **params)
    if name == "vgg":
        return VGGStyle(num_classes, **params)
    if name == "resnet_mini":
        return ResNetMini(num_classes, **params)
    if name == "resnet18":
        return build_resnet18(num_classes, **params)
    raise ValueError(f"Unknown model '{name}'. "
                     f"Choices: mlp, tiny_cnn, vgg, resnet_mini, resnet18")
