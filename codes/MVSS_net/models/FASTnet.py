''' Fuseformer for Video Inpainting
'''
import torch
import torch.nn as nn
from vit_pytorch import ViT


class AddPosEmb(nn.Module):
    def __init__(self):
        super(AddPosEmb, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, 196, 1024).float().normal_(mean=0, std=0.02), requires_grad=True)

    def forward(self, x):
        x = x + self.pos_emb
        return x


# 将每一个patch单独输入linear中并最终生成一个[1024,N]尺寸数据
class LinearEmbedding(nn.Module):
    def __init__(self, input, hidden, kernel_size, dilation, stride, padding, dropout=0.1):
        super(LinearEmbedding, self).__init__()
        # Unfold可理解为只卷不积，即提取kernel_size覆盖区域像素。dilation=1说明不采用空洞卷
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding)
        self.embedding1 = nn.Linear(input, hidden)  # spatial embedding
        self.embedding2 = nn.Linear(hidden * 3, hidden)  # temporal embedding
        self.embedding2_ = nn.Linear(196 * 3, 196)  # temporal embedding
        self.dropout = nn.Dropout(p=dropout)
        self.add_pos_emb = AddPosEmb()  # position embedding

    def forward(self, frames, b, t):
        x = self.unfold(frames)  # 图片分块，输入(N,C,W,H)->(b*t,3,224,224)，输出(b*t,16*16*3,14*14)->(b*t,768,196)
        x = x.permute(0, 2, 1)  # 将unfold结果后两维换位，(b*t,196,768)
        x = self.embedding1(x)  # spatial dimension，每帧单独通过linear得到embedding，输出(b*t,196,1024)
        x = self.dropout(x)
        x = x.view(b, t, 196, 1024)

        # x = x.permute(0, 2, 3, 1)  # (b,196,1024,t)
        # x = x.reshape(b, 196, 1024*t)  # (b,196,1024,t) -> (b,196,t*1024)=(b,196,3072)
        # x = self.embedding2(x)  # temporal dimension，输出(b,196,1024)
        x = x.reshape(b, t * 196, 1024)
        x = x.permute(0, 2, 1)  # (b,1024,t*196)
        x = self.embedding2_(x)  # temporal dimension，输出(b,1024,196)
        x = x.permute(0, 2, 1)  # (b,196,1024)

        x = self.dropout(x)
        x = self.add_pos_emb(x)  # 输出(b,n,c) -> (b,196,1024)=(b,196,1024)
        return x


class VIT(nn.Module):
    def __init__(self):
        super(VIT, self).__init__()
        self.v = ViT(
            image_size=224,  # Image size
            patch_size=16,  # Size of patches
            num_classes=1000,  # Number of classes to classify.
            dim=1024,  # Last dimension of output tensor after linear transformation
            depth=8,  # Number of Transformer blocks
            heads=16,  # Number of heads in Multi-head Attention layer
            mlp_dim=2048,  # Dimension of the MLP (FeedForward) layer
            channels=3,  # Number of image's channels. default 3
            dropout=0.1,  # Dropout rate. float between [0, 1], default 0
            emb_dropout=0.1,  # Embedding dropout rate. float between [0, 1], default 0
            pool='cls'  # string, either cls token pooling or mean pooling
        )
        # 加载预训练模型参数
        self.v.load_state_dict(torch.load("/home/wanghao/Project/FAST/model/imagenet21k+imagenet2012_ViT-B_16-224.pth"),
                               strict=False)
        # 去掉linear embedding层和最后一层classifier
        self.modified_model = nn.Sequential(*list(self.v.children())[1:-1])

    def forward(self, input):
        return self.modified_model(input)


class FreDowmsample(nn.Module):
    def __init__(self):
        super(FreDowmsample, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # input [b,3,224,224]
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # input [b,32,224,224] (H,W,D4)
        self.relu2 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # input [b,64,112,112] (H/2,W/2,D3)
        self.relu3 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # input [b,128,56,56]  (H/4,W/4,D2)
        self.relu4 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))                      # output [b,256,28,28]  (H/8,W/8,D1)

    def forward(self, input):
        # [b,3,224,224]
        x = self.conv1(input)
        fre_div1 = self.relu1(x)  # [b, 32, 224, 224] (H,W,D4)
        x = self.conv2(fre_div1)
        fre_div2 = self.relu2(x)
        # fre_div2 = self.pool1(x)    # [b, 64, 112, 112] (H/2,W/2,D3)
        x = self.conv3(fre_div2)
        fre_div4 = self.relu3(x)
        # fre_div4 = self.pool2(x)    # [b, 128, 56, 56]  (H/4,W/4,D2)
        x = self.conv4(fre_div4)
        fre_div8 = self.relu4(x)
        # fre_div8 = self.pool3(x)    # [b, 256, 28, 28]  (H/8,W/8,D1)
        return (fre_div1, fre_div2, fre_div4, fre_div8)


class FreUpsample(nn.Module):
    def __init__(self):
        super(FreUpsample, self).__init__()
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256 + 256, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128 + 128, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64 + 64, 32, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(32 + 32, 1, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, fre_div1, fre_div2, fre_div4, fre_div8, input):
        x = self.conv1(input)  # input [b,14,14,1024]
        x = self.relu1(x)
        x = self.upsample(x)  # output [b,256,28,28] (H/8,W/8,D1*2)
        concat1 = torch.cat([fre_div8, x], 1)  # output [b,256+256,28,28] (H/8,W/8,D1*2)

        x = self.conv2(concat1)
        x = self.relu2(x)
        x = self.upsample(x)  # output [b,128,56,56] (H/8,W/8,D1*2)
        concat2 = torch.cat([fre_div4, x], 1)

        x = self.conv3(concat2)  # input [128+128,56,56] (H/4,W/4,D2*2)
        x = self.relu3(x)
        x = self.upsample(x)  # output [b,64,112,112] (H/8,W/8,D1*2)
        concat3 = torch.cat([fre_div2, x], 1)

        x = self.conv4(concat3)  # input [64+64,112,112] (H/2,W/2,D3*2)
        x = self.relu4(x)
        x = self.upsample(x)  # output [b,32,224,224] (H/8,W/8,D1*2)
        concat4 = torch.cat([fre_div1, x], 1)  # output [32+32,224,224]  (H,W,D4*2)

        output = self.conv5(concat4)  # output [b,2,224,224]  (H,W,1)
        # output = self.relu5(output)           # 由于在计算损失中已经包含对pred结果进行sigmoid的操作，故在此可先注释掉
        return output


# class FreUpsample(nn.Module):
#     def __init__(self):
#         super(FreUpsample, self).__init__()
#         self.convtrans1 = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2)
#         self.relu1 = nn.ReLU()
#         self.convtrans2 = nn.ConvTranspose2d(256+256, 128, kernel_size=2, stride=2)
#         self.relu2 = nn.ReLU()
#         self.convtrans3 = nn.ConvTranspose2d(128+128, 64, kernel_size=2, stride=2)
#         self.relu3 = nn.ReLU()
#         self.convtrans4 = nn.ConvTranspose2d(64+64, 32, kernel_size=2, stride=2)
#         self.relu4 = nn.ReLU()
#         self.conv1 = nn.Conv2d(32+32, 1, kernel_size=3, stride=1, padding='same')
#         self.relu5 = nn.ReLU()


#     def forward(self, fre_div1, fre_div2, fre_div4, fre_div8, input):
#         x = self.convtrans1(input)              # input [b,14,14,1024]
#         # x = self.relu1(x)                       # output [b,256,28,28] (H/8,W/8,D1*2)
#         concat1 = torch.cat([fre_div8, x], 1)   # output [b,256+256,28,28] (H/8,W/8,D1*2)
#         x = self.convtrans2(concat1)
#         # x = self.relu2(x)
#         concat2 = torch.cat([fre_div4, x], 1)
#         x = self.convtrans3(concat2)            # input [128+128,56,56] (H/4,W/4,D2*2)
#         # x = self.relu3(x)
#         concat3 = torch.cat([fre_div2, x], 1)
#         x = self.convtrans4(concat3)            # input [64+64,112,112] (H/2,W/2,D3*2)
#         # x = self.relu4(x)
#         concat4 = torch.cat([fre_div1, x], 1)   # output [32+32,224,224]  (H,W,D4*2)
#         output = self.conv1(concat4)            # output [2,224,224]  (H,W,1)
#         # output = self.relu5(output)           # 由于在计算损失中已经包含对pred结果进行sigmoid的操作，故在此可先注释掉
#         return output

class InpaintGenerator(nn.Module):
    def __init__(self):
        super(InpaintGenerator, self).__init__()
        input = 768  # Linear embedding的输入
        hidden = 1024  # Linear embedding的输出
        kernel_size = (16, 16)  # Linear embedding的核大小
        dilation = (1, 1)  # (1,1) -> Linear embedding不采用空洞卷
        padding = (0, 0)  # Linear embedding的padding
        stride = (16, 16)  # Linear embedding的步长
        self.embedding = LinearEmbedding(input=input, hidden=hidden, kernel_size=kernel_size, dilation=dilation,
                                         stride=stride, padding=padding)
        self.transformer = VIT()  # VIT
        # (S,D0)降采样为(H/16,W/16,D0)
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu1 = nn.ReLU()
        self.downsample = FreDowmsample()  # 频谱图降采样encoder
        self.upsample = FreUpsample()  # 上采样decoder

    def forward(self, frames, spectrum):
        b, t, c, h, w = frames.size()
        frames = frames.view(b * t, c, h, w)
        _, c, h, w = frames.size()
        trans_feat = self.embedding(frames, b, t)  # output [b,196,1024]
        trans_feat = self.transformer(trans_feat)  # output [b,196,1024]
        trans_feat = trans_feat.view(b, 14, 14, trans_feat.shape[2])  # output [b,14,14,1024]
        trans_feat = trans_feat.permute(0, 3, 1, 2)  # output [b,1024,14,14]
        # trans_feat = self.conv1(trans_feat) # 这层卷积是在VIT的输出与上采样的输入之间
        # trans_feat = self.relu1(trans_feat)
        fre_div1, fre_div2, fre_div4, fre_div8 = self.downsample(spectrum)  # input [b,3,224,224] output [b,1024,14,14]
        output = self.upsample(fre_div1, fre_div2, fre_div4, fre_div8, trans_feat)  # output [b,1024,14,14]
        return output