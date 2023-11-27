import torch
from torch import nn


# Conv2d unit:
#       [Conv2d 5x5 -> BN -> ReLU] x 1
# ChannelNo: input_channel - > output_channel
# ImgSize: ImgSize -> ImgSize
class Conv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


# TranConv unit:
#       TransConv2d 5x5 -> BN -> ReLU
# ChannelNo: input_channel -> output_channel
# ImgSize: ImgSize -> ImgSize
class TranConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(TranConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


# TranConv residual unit:
#       TransConv2d 5x5 -> BN -> Sum -> ReLU
# ChannelNo: input_channel -> output_channel
# ImgSize: ImgSize -> ImgSize
class TranConv_Res(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(TranConv_Res, self).__init__()
        self.Tranconv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=5, stride=1, padding=2,
                                           bias=False)
        self.BN = nn.BatchNorm2d(output_channel)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, input, x):  # x: Input features from the corresponding SingleConv layer
        out = self.Tranconv(input)
        out = self.BN(out)
        out += x  # Summation for residual
        out = self.ReLU(out)
        return out


class REDCNN(nn.Module):
    def __init__(self):
        super(REDCNN, self).__init__()
        # Encoder
        self.Conv1 = Conv(1, 96)
        self.Conv2 = Conv(96, 96)
        self.Conv3 = Conv(96, 96)
        self.Conv4 = Conv(96, 96)
        self.Conv5 = Conv(96, 96)
        # Decoder
        self.Tranconv1 = TranConv_Res(96, 96)
        self.Tranconv2 = TranConv(96, 96)
        self.Tranconv3 = TranConv_Res(96, 96)
        self.Tranconv4 = TranConv(96, 96)
        self.Tranconv5 = nn.ConvTranspose2d(96, 1, kernel_size=5, stride=1, padding=2, bias=True)

        # Apply weight initialization to all modules
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # Kaiming (He)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c1 = self.Conv1(x)
        c2 = self.Conv2(c1)
        c3 = self.Conv3(c2)
        c4 = self.Conv4(c3)
        c5 = self.Conv5(c4)
        tc1 = self.Tranconv1(c5, c4)
        tc2 = self.Tranconv2(tc1)
        tc3 = self.Tranconv3(tc2, c2)
        tc4 = self.Tranconv4(tc3)
        out = self.Tranconv5(tc4)

        return out


if __name__ == '__main__':
    model = REDCNN()
    a = torch.randn((1, 1, 512, 512))
    o = model(a)
