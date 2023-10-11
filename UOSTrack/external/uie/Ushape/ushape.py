import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
import datetime
import os
import time
import timeit
import copy
import numpy as np
from torch.nn import ModuleList
from torch.nn import Conv2d
from torch.nn import LeakyReLU
from .block import *
from .block import _equalized_conv2d
from .SGFMT import TransformerModel
from .PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding
from .CMSFFT import ChannelTransformer
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from external.uie.FUnIE_GAN.img_utils import save_image


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self,
                 img_dim=256,
                 patch_dim=16,
                 embedding_dim=512,
                 num_channels=3,
                 num_heads=8,
                 num_layers=4,
                 hidden_dim=256,
                 dropout_rate=0.0,
                 attn_dropout_rate=0.0,
                 in_ch=3,
                 out_ch=3,
                 conv_patch_representation=True,
                 positional_encoding_type="learned",
                 use_eql=True):
        super(Generator, self).__init__()
        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.out_ch = out_ch  # 输出通道数
        self.in_ch = in_ch  # 输入通道数
        self.img_dim = img_dim  # 输入图片尺寸
        self.embedding_dim = embedding_dim  # 512
        self.num_heads = num_heads  # 多头注意力中头的数量
        self.patch_dim = patch_dim  # 每个patch的尺寸
        self.num_channels = num_channels  # 图片通道数?
        self.dropout_rate = dropout_rate  # drop-out比率
        self.attn_dropout_rate = attn_dropout_rate  # 注意力模块的dropout比率
        self.conv_patch_representation = conv_patch_representation  # True

        self.num_patches = int((img_dim // patch_dim) ** 2)  # 将三通道图片分成多少块
        self.seq_length = self.num_patches  # 每个sequence的长度为patches的大小
        self.flatten_dim = 128 * num_channels  # 128*3=384

        # 线性编码
        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        # 位置编码
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,  # 512
            num_layers,  # 4
            num_heads,  # 8
            hidden_dim,  # 4096

            self.dropout_rate,
            self.attn_dropout_rate,
        )

        # layer Norm
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:
            self.Conv_x = nn.Conv2d(
                256,
                self.embedding_dim,  # 512
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        # modulelist
        self.rgb_to_feature = ModuleList([from_rgb(32), from_rgb(64), from_rgb(128)])
        self.feature_to_rgb = ModuleList([to_rgb(32), to_rgb(64), to_rgb(128), to_rgb(256)])

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(self.in_ch, 16)
        self.Conv1_1 = conv_block(16, 32)
        self.Conv2 = conv_block(32, 32)
        self.Conv2_1 = conv_block(32, 64)
        self.Conv3 = conv_block(64, 64)
        self.Conv3_1 = conv_block(64, 128)
        self.Conv4 = conv_block(128, 128)
        self.Conv4_1 = conv_block(128, 256)

        self.Conv5 = conv_block(512, 256)

        # self.Conv_x = conv_block(256,512)
        self.mtc = ChannelTransformer(channel_num=[32, 64, 128, 256],
                                      patchSize=[32, 16, 8, 4])

        self.Up5 = up_conv(256, 256)
        self.coatt5 = CCA(F_g=256, F_x=256)
        self.Up_conv5 = conv_block(512, 256)
        self.Up_conv5_1 = conv_block(256, 256)

        self.Up4 = up_conv(256, 128)
        self.coatt4 = CCA(F_g=128, F_x=128)
        self.Up_conv4 = conv_block(256, 128)
        self.Up_conv4_1 = conv_block(128, 128)

        self.Up3 = up_conv(128, 64)
        self.coatt3 = CCA(F_g=64, F_x=64)
        self.Up_conv3 = conv_block(128, 64)
        self.Up_conv3_1 = conv_block(64, 64)

        self.Up2 = up_conv(64, 32)
        self.coatt2 = CCA(F_g=32, F_x=32)
        self.Up_conv2 = conv_block(64, 32)
        self.Up_conv2_1 = conv_block(32, 32)

        self.Conv = nn.Conv2d(32, self.out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()
    #
    def reshape_output(self, x):  # 将transformer的输出resize为原来的特征图尺寸
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )  # B,16,16,512
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

    def forward(self, x):
        # print(x.shape)

        output = []

        x_1 = self.Maxpool(x)
        x_2 = self.Maxpool(x_1)
        x_3 = self.Maxpool(x_2)

        e1 = self.Conv1(x)
        # print(e1.shape)
        e1 = self.Conv1_1(e1)
        e2 = self.Maxpool1(e1)
        # 32*128*128

        x_1 = self.rgb_to_feature[0](x_1)
        # e2=torch.cat((x_1,e2), dim=1)
        e2 = x_1 + e2
        e2 = self.Conv2(e2)
        e2 = self.Conv2_1(e2)
        e3 = self.Maxpool2(e2)
        # 64*64*64

        x_2 = self.rgb_to_feature[1](x_2)
        # e3=torch.cat((x_2,e3), dim=1)
        e3 = x_2 + e3
        e3 = self.Conv3(e3)
        e3 = self.Conv3_1(e3)
        e4 = self.Maxpool3(e3)
        # 128*32*32

        x_3 = self.rgb_to_feature[2](x_3)
        # e4=torch.cat((x_3,e4), dim=1)
        e4 = x_3 + e4
        e4 = self.Conv4(e4)
        e4 = self.Conv4_1(e4)
        e5 = self.Maxpool4(e4)
        # 256*16*16

        # channel-wise transformer-based attention
        e1, e2, e3, e4, att_weights = self.mtc(e1, e2, e3, e4)

        # spatial-wise transformer-based attention
        residual = e5
        # 中间的隐变量
        # conv_x应该接受256通道，输出512通道的中间隐变量
        e5 = self.bn(e5)
        e5 = self.relu(e5)
        e5 = self.Conv_x(e5)  # out->512*16*16 shape->B,512,16,16
        e5 = e5.permute(0, 2, 3, 1).contiguous()  # B,512,16,16->B,16,16,512
        e5 = e5.view(e5.size(0), -1, self.embedding_dim)  # B,16,16,512->B,16*16,512 线性映射层
        e5 = self.position_encoding(e5)  # 位置编码
        e5 = self.pe_dropout(e5)  # 预dropout层
        # apply transformer
        e5 = self.transformer(e5)
        e5 = self.pre_head_ln(e5)
        e5 = self.reshape_output(e5)  # out->512*16*16 shape->B,512,16,16
        e5 = self.Conv5(e5)  # out->256,16,16 shape->B,256,16,16
        # residual是否要加bn和relu？
        e5 = e5 + residual

        d5 = self.Up5(e5)
        e4_att = self.coatt5(g=d5, x=e4)
        d5 = torch.cat((e4_att, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d5 = self.Up_conv5_1(d5)
        # 256
        out3 = self.feature_to_rgb[3](d5)
        output.append(out3)  # 32*32orH/8,W/8

        d4 = self.Up4(d5)
        e3_att = self.coatt4(g=d4, x=e3)
        d4 = torch.cat((e3_att, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.Up_conv4_1(d4)
        # 128
        out2 = self.feature_to_rgb[2](d4)
        output.append(out2)  # 64*64orH/4,W/4

        d3 = self.Up3(d4)
        e2_att = self.coatt3(g=d3, x=e2)
        d3 = torch.cat((e2_att, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.Up_conv3_1(d3)
        # 64
        out1 = self.feature_to_rgb[1](d3)
        output.append(out1)  # 128#128orH/2,W/2

        d2 = self.Up2(d3)
        e1_att = self.coatt2(g=d2, x=e1)
        d2 = torch.cat((e1_att, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.Up_conv2_1(d2)
        # 32
        out0 = self.feature_to_rgb[0](d2)
        output.append(out0)  # 256*256

        # out = self.Conv(d2)

        # d1 = self.active(out)
        # output=np.array(output)

        return output


class uhead(nn.Module):
    def __init__(self, model):
        super(uhead, self).__init__()
        self.model = model.cuda()
        self.transform = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC),
                                             transforms.ToTensor(), ])

    def forward(self, x):
        H, W, _ = x.shape
        x = Image.fromarray(x)
        x = self.transform(x)
        x = Variable(x).cuda().unsqueeze(0) / 255.
        with torch.no_grad():
            x = self.model(x)[3].data
        x = cv2.resize(save_image(x, nrow=5), (W, H))
        return x


def build_ushape():
    model = Generator().eval()
    pretrained = r'../external/uie/Ushape/generator_795.pth'
    model.load_state_dict(torch.load(pretrained))
    return uhead(model)
