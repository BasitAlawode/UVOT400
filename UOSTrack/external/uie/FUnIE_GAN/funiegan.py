"""
 > Network architecture of FUnIE-GAN model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from UOSTrack.external.uie.FUnIE_GAN.img_utils import save_image

is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorFunieGAN(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """

    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorFunieGAN, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        # decoding layers
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        return self.final(u45)


class uhead(nn.Module):
    def __init__(self, model):
        super(uhead, self).__init__()
        self.model = model.cuda()
        self.transform = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    def forward(self, x):
        H,W,_ = x.shape
        x = Image.fromarray(x)
        x = self.transform(x)
        x = Variable(x).cuda().unsqueeze(0)
        with torch.no_grad():
            x = self.model(x)
        x = cv2.resize(save_image(x), (W, H))
        return x


def build_fuinegan():
    model = GeneratorFunieGAN().eval()
    #pretrained = r'../external/uie/FUnIE_GAN/funie_generator.pth'
    pretrained = r'trained_trackers/uostrack/funie_generator.pth'
    model.load_state_dict(torch.load(pretrained), strict=True)
    return uhead(model)


if __name__ == '__main__':
    a = torch.rand(1, 3, 256, 256)
    model = GeneratorFunieGAN()
    b = model(a)
    print(b.shape)
