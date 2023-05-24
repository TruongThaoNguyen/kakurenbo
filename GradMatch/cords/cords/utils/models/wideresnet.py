import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import param_init

class BaseModel(nn.Module):
    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                f = self.feature_extractor(x)
                f = f.mean((2, 3))
        else:
            f = self.feature_extractor(x)
            f = f.mean((2, 3))
        if last:
            return self.classifier(f), f
        else:
            return self.classifier(f)

    def logits_with_feature(self, x):
        f = self.feature_extractor(x)
        c = self.classifier(f.mean((2, 3)))
        return c, f

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


def conv3x3(i_c, o_c, stride=1, bias=False):
    return nn.Conv2d(i_c, o_c, 3, stride, 1, bias=bias)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, channels, momentum=1e-3, eps=1e-3):
        super().__init__(channels)
        self.update_batch_stats = True

    def forward(self, x):
        if self.update_batch_stats or not self.training:
            return super().forward(x)
        else:
            return nn.functional.batch_norm(
                x, None, None, self.weight, self.bias, True, self.momentum, self.eps
            )


def leaky_relu():
    return nn.LeakyReLU(0.1)


class _Residual(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, activate_before_residual=False):
        super().__init__()
        layer = []
        if activate_before_residual:
            self.pre_act = nn.Sequential(
                BatchNorm2d(input_channels),
                leaky_relu()
            )
        else:
            self.pre_act = nn.Identity()
            layer.append(BatchNorm2d(input_channels))
            layer.append(leaky_relu())
        layer.append(conv3x3(input_channels, output_channels, stride))
        layer.append(BatchNorm2d(output_channels))
        layer.append(leaky_relu())
        layer.append(conv3x3(output_channels, output_channels))

        if stride >= 2 or input_channels != output_channels:
            self.identity = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
        else:
            self.identity = nn.Identity()

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        x = self.pre_act(x)
        return self.identity(x) + self.layer(x)


class WideResNet(BaseModel):
    """
    ResNet

    Parameters
    --------
    num_classes: int
        number of classes
    filters: int
        number of filters
    scales: int
        number of scales
    repeat: int
        number of residual blocks per scale
    dropout: float
        dropout ratio (None indicates dropout is unused)
    """
    def __init__(self, num_classes, filters, scales, repeat, dropout=None, *args, **kwargs):
        super().__init__()
        feature_extractor = [conv3x3(3, 16)]
        channels = 16
        for scale in range(scales):
            feature_extractor.append(
                _Residual(channels, filters<<scale, 2 if scale else 1, activate_before_residual = (scale == 0))
            )
            channels = filters << scale
            for _ in range(repeat - 1):
                feature_extractor.append(
                    _Residual(channels, channels)
                )

        feature_extractor.append(BatchNorm2d(channels))
        feature_extractor.append(leaky_relu())
        self.feature_extractor = nn.Sequential(*feature_extractor)
        classifier = []
        if dropout is not None:
            classifier.append(nn.Dropout(dropout))
        classifier.append(nn.Linear(channels, num_classes))
        self.embDim = channels
        self.classifier = nn.Sequential(*classifier)
        param_init(self.modules())

    def get_embedding_dim(self):
        return self.embDim
