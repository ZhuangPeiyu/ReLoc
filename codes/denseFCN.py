import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torchvision.models.resnet import BasicBlock,ResNet
from torchvision.models.densenet import _densenet
import numpy as np
from models.senet import se_resnext50_32x4d, se_resnext101_32x4d, SCSEModule, senet154
# from networks import highPassingFilters
# import pytorch_colors as colors #pytorch颜色空间转换包
# 2xpadding-dilation*(k-1) = 0
class ConvX(nn.Module):
    def __init__(self,in_channels,filters,kernel_size,strides,padding,weight_decay,bn_in,dilate_rate,is_training):
        super(ConvX,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=padding,dilation=dilate_rate)
        self.bn_in = bn_in
        if (self.bn_in == 'bn'):
            self.bn_layer = nn.BatchNorm2d(num_features=filters,affine=True)
        if (self.bn_in == 'in'):
            self.in_layer = nn.InstanceNorm2d(num_features=filters,affine = True)
        self.act_layer = nn.ReLU(inplace=True) #不保存中间变量

    def forward(self,x):
        x = self.conv(x)
        if(self.bn_in =='bn'):
            x = self.bn_layer(x)
        if(self.bn_in == 'in'):
            x = self.in_layer(x)
        x = self.act_layer(x)
        return x

class dense_block(nn.Module):
    def __init__(self,in_channels,num_conv,kernel_size,filters,output_channels,dilate_rate,weight_decay,name,down_sample,is_training,bn_in,strides,padding):
        super(dense_block, self).__init__()
        self.num_conv = num_conv
        # if(self.num_conv==2):
        #     self.conv1 = ConvX(in_channels,filters,kernel_size,strides,padding,weight_decay,bn_in,dilate_rate,is_training)
        #     self.conv2 = ConvX(in_channels+filters,filters,kernel_size,strides,padding,weight_decay,bn_in,dilate_rate,is_training)
        # if(self.num_conv==4):
        #     self.conv1 = ConvX(in_channels, filters, kernel_size, strides, padding, weight_decay, bn_in, dilate_rate,
        #                        is_training)
        #     self.conv2 = ConvX(in_channels + filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
        #                        dilate_rate, is_training)
        #     self.conv3 = ConvX(in_channels+2*filters,filters,kernel_size,strides,padding,weight_decay,bn_in,dilate_rate,is_training)
        #     self.conv4 = ConvX(in_channels + 3 * filters, filters, kernel_size, strides, padding, weight_decay, bn_in,dilate_rate, is_training)
        self.conv1 = ConvX(in_channels, filters, kernel_size, strides, padding, weight_decay, bn_in, dilate_rate,
                           is_training)
        self.conv2 = ConvX(in_channels + filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
                           dilate_rate, is_training)
        self.conv3 = ConvX(in_channels + 2 * filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
                           dilate_rate, is_training)
        self.conv4 = ConvX(in_channels + 3 * filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
                           dilate_rate, is_training)
        self.down_sample = down_sample
        if(self.num_conv==2):
            self.transition_layer = nn.Conv2d(in_channels=in_channels + 2 * filters, out_channels=output_channels, kernel_size=kernel_size, stride=strides, padding=padding,dilation=dilate_rate)
        if (self.num_conv == 3):
            self.transition_layer = nn.Conv2d(in_channels=in_channels + 3 * filters, out_channels=output_channels,
                                              kernel_size=kernel_size, stride=strides, padding=padding,
                                              dilation=dilate_rate)
        if (self.num_conv == 4):
            self.transition_layer = nn.Conv2d(in_channels=in_channels + 4 * filters, out_channels=output_channels,
                                              kernel_size=kernel_size, stride=strides, padding=padding,
                                              dilation=dilate_rate)
        self.se_layer = ChannelSELayer(output_channels,reduction_ratio=2)
    def forward(self,x):
        if(self.num_conv==2):
            conv1_output = self.conv1(x)
            conv2_input = torch.cat([x,conv1_output],dim = 1)
            conv2_output = self.conv2(conv2_input)
            transition_input = torch.cat([x,conv1_output,conv2_output],dim = 1)
            x = self.transition_layer(transition_input)
            if (self.down_sample == True):
                x = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)(x)
            x = self.se_layer(x)
            return x
        if (self.num_conv == 3):
            conv1_output = self.conv1(x)
            conv2_input = torch.cat([x, conv1_output], dim=1)
            conv2_output = self.conv2(conv2_input)
            conv3_input = torch.cat([x, conv1_output, conv2_output], dim=1)
            conv3_output = self.conv3(conv3_input)
            transition_input = torch.cat([x, conv1_output, conv2_output, conv3_output], dim=1)
            x = self.transition_layer(transition_input)
            if (self.down_sample == True):
                x = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)(x)
            x = self.se_layer(x)
            return x
        if(self.num_conv==4):
            conv1_output = self.conv1(x)
            conv2_input = torch.cat([x,conv1_output],dim = 1)
            conv2_output = self.conv2(conv2_input)
            conv3_input = torch.cat([x,conv1_output,conv2_output],dim = 1)
            conv3_output = self.conv3(conv3_input)
            conv4_input = torch.cat([x,conv1_output,conv2_output,conv3_output],dim = 1)
            conv4_output = self.conv4(conv4_input)
            transition_input = torch.cat([x,conv1_output,conv2_output,conv3_output,conv4_output],dim = 1)
            x = self.transition_layer(transition_input)
            if(self.down_sample==True):
                x = nn.AvgPool2d(kernel_size=2,stride = 2,padding=0)(x)
            x = self.se_layer(x)
            return x


def make_DCT_filter_anysize(win_size):
    DCT_filter_n = np.zeros([win_size, win_size, 1, win_size * win_size])

    XX, YY = np.meshgrid(range(win_size), range(win_size))

    C = np.ones(win_size)
    C[0] = np.sqrt(1 / win_size)
    C[1:] = np.sqrt(2 / win_size)
    for v in range(win_size):
        for u in range(win_size):
            DCT_filter_n[:, :, 0, u + v * win_size] = C[v] * C[u] * np.cos(
                (YY + 0.5) * np.pi * v / win_size) * np.cos((XX + 0.5) * np.pi * u / win_size)
    print("DCT_filter_n: ",DCT_filter_n.shape)
    DCT_filter_n = DCT_filter_n.transpose(3,2,0,1)
    DCT_filter = torch.from_numpy(DCT_filter_n.astype(np.float32))

    return DCT_filter

def conv2d_block(x, W, overlapping_step,padding = 0):
    W.requires_grad = False
    return torch.nn.functional.conv2d(x,W,bias = None,stride=(overlapping_step,overlapping_step),padding = 0) #这个在这里存疑

class dct_transform_net(nn.Module):
    def __init__(self,win_size,overlapping_step):
        super(dct_transform_net, self).__init__()
        self.DCT_filter = make_DCT_filter_anysize(win_size)
        self.win_size = win_size
        self.overlapping_step = overlapping_step

    def forward(self,x):
        x = conv2d_block(x,self.DCT_filter,self.overlapping_step)
        return x
# class SE_Block(nn.Module):
#     def __init__(self, ch_in, reduction=16):
#         super(SE_Block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化,output feature：[batch_size,in_channel,1,1]
#         self.fc = nn.Sequential(
#             nn.Linear(ch_in, ch_in // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(ch_in // reduction, ch_in, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x),y.expand_as(x)
class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):

        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4,bn_in = 'in'):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        if(bn_in=='bn'):
            # print("BNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
        if (bn_in == 'in'):
            # print("INXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(inter_channels,affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(channels,affine=True),
            )

            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(inter_channels,affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(channels,affine=True),
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        print("xa shape: ",xa.shape,'xl shape: ',xl.shape)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        # print("xo shape: ",xo.shape)
        return xo

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
  if transposed:
    layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=stride-1, dilation=dilation, bias=bias)

    # layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=0, dilation=dilation, bias=bias)
    # Bilinear interpolation init 用双线性插值法初始化反卷积核
    w = torch.Tensor(kernel_size, kernel_size)
    centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
    for y in range(kernel_size):
      for x in range(kernel_size):
        w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs((y - centre) / stride))
    layer.weight.data.copy_(w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
  else:
    padding = (kernel_size + 2 * (dilation - 1)) // 2
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
  if bias:
    init.constant(layer.bias, 0)
  return layer

class ConvSCSEBottleneckNoBn(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2):
        # print('bottleneck ', in_channels, out_channels)
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            SCSEModule(out_channels, reduction=reduction, mode='maxout'),
        )

    def forward(self, dec, enc):
        # print(dec.shape,enc.shape)
        # enc = torch.nn.functional.interpolate(enc,
        #                                 size=(dec.shape[2], dec.shape[3]))
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)

class normal_denseFCN(nn.Module):
    def __init__(self, bn_in,return_middle_map = False):
        super(normal_denseFCN, self).__init__()
        self.return_middle_map = return_middle_map
        self.dense_block1 = dense_block(in_channels=3, num_conv=4, kernel_size=3, filters=8, output_channels=16,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)


        self.dense_block2 = dense_block(in_channels=16, num_conv=2, kernel_size=3, filters=16, output_channels=32,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        self.dense_block3 = dense_block(in_channels=32, num_conv=2, kernel_size=3, filters=32, output_channels=64,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        self.dense_block4 = dense_block(in_channels=64, num_conv=2, kernel_size=3, filters=64, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.dense_block5 = dense_block(in_channels=96, num_conv=2, kernel_size=3, filters=96, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.de_conv1 = ConvX(in_channels=96, filters=64, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        self.dense_block6 = dense_block(in_channels=96, num_conv=2, kernel_size=3, filters=32, output_channels=64,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)
        # self.de_conv2 = conv(in_planes = 64+32, out_planes = 48, kernel_size=4, stride=2, dilation=1, bias=False, transposed=True)
        self.de_conv2 = ConvX(in_channels=64, filters=48, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        self.dense_block7 = dense_block(in_channels=64, num_conv=2, kernel_size=3, filters=32, output_channels=48,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        # self.de_conv3 = conv(in_planes=48+16, out_planes=32, kernel_size=4, stride=2, dilation=1, bias=False,
        #                      transposed=True)
        self.de_conv3 = ConvX(in_channels=48, filters=32, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)

        # self.final_conv = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=5, stride=1, padding=2)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        processed_image = x
        spatial_input = processed_image
        '''spatial branch'''
        spatial_dense_block1 = self.dense_block1(spatial_input)  # 空域第一个dense block的输出
        spatial_dense_block2 = self.dense_block2(spatial_dense_block1)
        spatial_dense_block3 = self.dense_block3(spatial_dense_block2)
        spatial_dense_block4 = self.dense_block4(spatial_dense_block3)
        spatial_dense_block5 = self.dense_block5(spatial_dense_block4)
        de_conv1_input = torch.nn.functional.interpolate(spatial_dense_block5,
                                                         size=(spatial_dense_block2.shape[2], spatial_dense_block2.shape[3]))

        de_conv1 = self.de_conv1(de_conv1_input)
        de_conv1 = torch.cat([de_conv1,spatial_dense_block2],dim = 1)
        spatial_dense_block6 = self.dense_block6(de_conv1)
        de_conv2_input = torch.nn.functional.interpolate(spatial_dense_block6,
                                                         size=(spatial_dense_block1.shape[2], spatial_dense_block1.shape[3]))
        de_conv2 = self.de_conv2(de_conv2_input)
        de_conv2 = torch.cat([de_conv2, spatial_dense_block1], dim=1)
        spatial_dense_block7 = self.dense_block7(de_conv2)
        de_conv3_input = torch.nn.functional.interpolate(spatial_dense_block7,
                                                         size=(spatial_input.shape[2], spatial_input.shape[3]))

        de_conv3 = self.de_conv3(de_conv3_input)
        logit_msk_output = self.final_conv(de_conv3)
        logit_msk_output = torch.sigmoid(logit_msk_output)

        # preds_msk_map = torch.softmax(logit_msk_output, dim=1)[:, 1, :, :]
        # preds_msk = torch.greater_equal(preds_msk_map, 0.5).to(torch.int32)
        # print("logit_msk_shape",logit_msk_output.shape)
        # return logit_msk_output,preds_msk_map,preds_msk
        # print("x shape: ",x.shape,"outpu shape : ",logit_msk_output.shape)
        if(self.return_middle_map):
            # return spatial_dense_block1,spatial_dense_block2,spatial_dense_block3,spatial_dense_block4,spatial_dense_block5,\
            #        spatial_dense_block6,spatial_dense_block7,de_conv1,de_conv2,de_conv3,logit_msk_output
            return spatial_dense_block5,logit_msk_output
        else:
            return logit_msk_output


class normal_denseFCN_deeper_small_res(nn.Module):
    def __init__(self, bn_in,return_middle_map = False):
        super(normal_denseFCN, self).__init__()
        self.return_middle_map = return_middle_map
        self.dense_block1 = dense_block(in_channels=3, num_conv=4, kernel_size=3, filters=8, output_channels=16,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)


        self.dense_block2 = dense_block(in_channels=16, num_conv=4, kernel_size=3, filters=16, output_channels=32,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        self.dense_block3 = dense_block(in_channels=32, num_conv=4, kernel_size=3, filters=32, output_channels=64,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        self.dense_block4 = dense_block(in_channels=64, num_conv=4, kernel_size=3, filters=64, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.dense_block5 = dense_block(in_channels=96, num_conv=4, kernel_size=3, filters=96, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.dense_block5_1 = dense_block(in_channels=96, num_conv=4, kernel_size=3, filters=128, output_channels=128,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.dense_block5_2 = dense_block(in_channels=128, num_conv=4, kernel_size=3, filters=196, output_channels=196,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.de_conv1 = ConvX(in_channels=96, filters=64, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        self.dense_block6 = dense_block(in_channels=96, num_conv=4, kernel_size=3, filters=32, output_channels=64,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)
        # self.de_conv2 = conv(in_planes = 64+32, out_planes = 48, kernel_size=4, stride=2, dilation=1, bias=False, transposed=True)
        self.de_conv2 = ConvX(in_channels=64, filters=48, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        self.dense_block7 = dense_block(in_channels=64, num_conv=4, kernel_size=3, filters=32, output_channels=48,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        # self.de_conv3 = conv(in_planes=48+16, out_planes=32, kernel_size=4, stride=2, dilation=1, bias=False,
        #                      transposed=True)
        self.de_conv3 = ConvX(in_channels=48, filters=32, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        self.dense_block8 = dense_block(in_channels=64, num_conv=4, kernel_size=3, filters=32, output_channels=48,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)
        self.de_conv4 = ConvX(in_channels=48, filters=32, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        # self.final_conv = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=5, stride=1, padding=2)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        processed_image = x
        spatial_input = processed_image
        '''spatial branch'''
        spatial_dense_block1 = self.dense_block1(spatial_input)  # 空域第一个dense block的输出
        spatial_dense_block2 = self.dense_block2(spatial_dense_block1)
        spatial_dense_block3 = self.dense_block3(spatial_dense_block2)
        spatial_dense_block4 = self.dense_block4(spatial_dense_block3)
        spatial_dense_block5 = self.dense_block5(spatial_dense_block4)
        spatial_dense_block5_1 = self.dense_block5_1(spatial_dense_block5)
        spatial_dense_block5_2 = self.dense_block5_2(spatial_dense_block5_1)

        de_conv1_input = torch.nn.functional.interpolate(spatial_dense_block5_2,
                                                         size=(spatial_dense_block3.shape[2], spatial_dense_block3.shape[3]))

        de_conv1 = self.de_conv1(de_conv1_input)
        de_conv1 = torch.cat([de_conv1,spatial_dense_block5],dim = 1)
        spatial_dense_block6 = self.dense_block6(de_conv1)
        de_conv2_input = torch.nn.functional.interpolate(spatial_dense_block6,
                                                         size=(spatial_dense_block1.shape[2], spatial_dense_block1.shape[3]))
        de_conv2 = self.de_conv2(de_conv2_input)
        de_conv2 = torch.cat([de_conv2, spatial_dense_block1], dim=1)
        spatial_dense_block7 = self.dense_block7(de_conv2)
        de_conv3_input = torch.nn.functional.interpolate(spatial_dense_block7,
                                                         size=(spatial_input.shape[2], spatial_input.shape[3]))

        de_conv3 = self.de_conv3(de_conv3_input)
        logit_msk_output = self.final_conv(de_conv3)
        logit_msk_output = torch.sigmoid(logit_msk_output)

        # preds_msk_map = torch.softmax(logit_msk_output, dim=1)[:, 1, :, :]
        # preds_msk = torch.greater_equal(preds_msk_map, 0.5).to(torch.int32)
        # print("logit_msk_shape",logit_msk_output.shape)
        # return logit_msk_output,preds_msk_map,preds_msk
        # print("x shape: ",x.shape,"outpu shape : ",logit_msk_output.shape)
        if(self.return_middle_map):
            return spatial_dense_block1,spatial_dense_block2,spatial_dense_block3,spatial_dense_block4,spatial_dense_block5,\
                   spatial_dense_block6,spatial_dense_block7,de_conv1,de_conv2,de_conv3,logit_msk_output
        else:
            return logit_msk_output

class deeper_denseFCN(nn.Module):
    def __init__(self, bn_in,return_middle_map = False):
        super(deeper_denseFCN, self).__init__()
        self.return_middle_map = return_middle_map
        self.dense_block1 = dense_block(in_channels=3, num_conv=4, kernel_size=3, filters=8, output_channels=16,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)


        self.dense_block2 = dense_block(in_channels=16, num_conv=4, kernel_size=3, filters=16, output_channels=32,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)

        self.dense_block3 = dense_block(in_channels=32, num_conv=4, kernel_size=3, filters=32, output_channels=64,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)

        self.dense_block4 = dense_block(in_channels=64, num_conv=4, kernel_size=3, filters=64, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.dense_block5 = dense_block(in_channels=96, num_conv=4, kernel_size=3, filters=96, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.de_conv1 = ConvX(in_channels=96, filters=64, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        self.dense_block6 = dense_block(in_channels=96, num_conv=2, kernel_size=3, filters=32, output_channels=64,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)
        # self.de_conv2 = conv(in_planes = 64+32, out_planes = 48, kernel_size=4, stride=2, dilation=1, bias=False, transposed=True)
        self.de_conv2 = ConvX(in_channels=64, filters=48, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        self.dense_block7 = dense_block(in_channels=64, num_conv=2, kernel_size=3, filters=32, output_channels=48,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        # self.de_conv3 = conv(in_planes=48+16, out_planes=32, kernel_size=4, stride=2, dilation=1, bias=False,
        #                      transposed=True)
        self.de_conv3 = ConvX(in_channels=48, filters=32, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        self.se_1 = ConvSCSEBottleneckNoBn(in_channels=96,out_channels=96)
        self.se_2 = ConvSCSEBottleneckNoBn(in_channels=64, out_channels=64)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        processed_image = x
        spatial_input = processed_image
        '''spatial branch'''
        spatial_dense_block1 = self.dense_block1(spatial_input)  # 空域第一个dense block的输出
        spatial_dense_block2 = self.dense_block2(spatial_dense_block1)
        spatial_dense_block3 = self.dense_block3(spatial_dense_block2)
        spatial_dense_block4 = self.dense_block4(spatial_dense_block3)
        spatial_dense_block5 = self.dense_block5(spatial_dense_block4)
        de_conv1_input = torch.nn.functional.interpolate(spatial_dense_block5,
                                                         size=(spatial_dense_block2.shape[2], spatial_dense_block2.shape[3]))

        de_conv1 = self.de_conv1(de_conv1_input)
        # de_conv1 = torch.cat([de_conv1,spatial_dense_block2],dim = 1)
        de_conv1 = self.se_1(de_conv1,spatial_dense_block2)
        spatial_dense_block6 = self.dense_block6(de_conv1)
        de_conv2_input = torch.nn.functional.interpolate(spatial_dense_block6,
                                                         size=(spatial_dense_block1.shape[2], spatial_dense_block1.shape[3]))
        de_conv2 = self.de_conv2(de_conv2_input)
        # de_conv2 = torch.cat([de_conv2, spatial_dense_block1], dim=1)
        de_conv2 = self.se_2(de_conv2,spatial_dense_block1)

        spatial_dense_block7 = self.dense_block7(de_conv2)
        de_conv3_input = torch.nn.functional.interpolate(spatial_dense_block7,
                                                         size=(spatial_input.shape[2], spatial_input.shape[3]))

        de_conv3 = self.de_conv3(de_conv3_input)
        logit_msk_output = self.final_conv(de_conv3)
        logit_msk_output = torch.sigmoid(logit_msk_output)
        # preds_msk_map = torch.softmax(logit_msk_output, dim=1)[:, 1, :, :]
        # preds_msk = torch.greater_equal(preds_msk_map, 0.5).to(torch.int32)
        # print("logit_msk_shape",logit_msk_output.shape)
        # return logit_msk_output,preds_msk_map,preds_msk
        # print("x shape: ",x.shape,"outpu shape : ",logit_msk_output.shape)
        if(self.return_middle_map):
            return spatial_dense_block1,spatial_dense_block2,spatial_dense_block3,spatial_dense_block4,spatial_dense_block5,\
                   spatial_dense_block6,spatial_dense_block7,de_conv1,de_conv2,de_conv3,logit_msk_output
        else:
            return logit_msk_output

class normal_denseFCN_deeper(nn.Module):
    def __init__(self, bn_in):
        super(normal_denseFCN_deeper, self).__init__()
        # self.dense_block1 = dense_block(in_channels=3, num_conv=4, kernel_size=3, filters=8, output_channels=16,
        #                                 dilate_rate=3, weight_decay=0, name='', down_sample=True, is_training=True,
        #                                 bn_in=bn_in, strides=1, padding=3)
        #
        #
        # self.dense_block2 = dense_block(in_channels=16, num_conv=4, kernel_size=3, filters=16, output_channels=32,
        #                                 dilate_rate=3, weight_decay=0, name='', down_sample=True, is_training=True,
        #                                 bn_in=bn_in, strides=1, padding=3)
        #
        # self.dense_block3 = dense_block(in_channels=32, num_conv=4, kernel_size=3, filters=32, output_channels=64,
        #                                 dilate_rate=3, weight_decay=0, name='', down_sample=True, is_training=True,
        #                                 bn_in=bn_in, strides=1, padding=3)
        self.dense_block1 = dense_block(in_channels=3, num_conv=4, kernel_size=3, filters=8, output_channels=16,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)


        self.dense_block2 = dense_block(in_channels=16, num_conv=4, kernel_size=3, filters=16, output_channels=32,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        self.dense_block3 = dense_block(in_channels=32, num_conv=4, kernel_size=3, filters=32, output_channels=64,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        self.dense_block4 = dense_block(in_channels=64, num_conv=4, kernel_size=3, filters=64, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.dense_block5 = dense_block(in_channels=96, num_conv=4, kernel_size=3, filters=96, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.de_conv1 = ConvX(in_channels=96, filters=64, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        self.dense_block6 = dense_block(in_channels=96, num_conv=4, kernel_size=3, filters=32, output_channels=64,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)
        # self.de_conv2 = conv(in_planes = 64+32, out_planes = 48, kernel_size=4, stride=2, dilation=1, bias=False, transposed=True)
        self.de_conv2 = ConvX(in_channels=64, filters=48, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        self.dense_block7 = dense_block(in_channels=64, num_conv=4, kernel_size=3, filters=32, output_channels=48,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        # self.de_conv3 = conv(in_planes=48+16, out_planes=32, kernel_size=4, stride=2, dilation=1, bias=False,
        #                      transposed=True)
        self.de_conv3 = ConvX(in_channels=48, filters=32, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)

        # self.final_conv = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=5, stride=1, padding=2)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        processed_image = x
        spatial_input = processed_image
        '''spatial branch'''
        spatial_dense_block1 = self.dense_block1(spatial_input)  # 空域第一个dense block的输出
        spatial_dense_block2 = self.dense_block2(spatial_dense_block1)
        spatial_dense_block3 = self.dense_block3(spatial_dense_block2)
        spatial_dense_block4 = self.dense_block4(spatial_dense_block3)
        spatial_dense_block5 = self.dense_block5(spatial_dense_block4)
        de_conv1_input = torch.nn.functional.interpolate(spatial_dense_block5,
                                                         size=(spatial_dense_block2.shape[2], spatial_dense_block2.shape[3]))

        de_conv1 = self.de_conv1(de_conv1_input)
        de_conv1 = torch.cat([de_conv1,spatial_dense_block2],dim = 1)
        spatial_dense_block6 = self.dense_block6(de_conv1)
        de_conv2_input = torch.nn.functional.interpolate(spatial_dense_block6,
                                                         size=(spatial_dense_block1.shape[2], spatial_dense_block1.shape[3]))
        de_conv2 = self.de_conv2(de_conv2_input)
        de_conv2 = torch.cat([de_conv2, spatial_dense_block1], dim=1)
        spatial_dense_block7 = self.dense_block7(de_conv2)
        de_conv3_input = torch.nn.functional.interpolate(spatial_dense_block7,
                                                         size=(spatial_input.shape[2], spatial_input.shape[3]))

        de_conv3 = self.de_conv3(de_conv3_input)
        logit_msk_output = self.final_conv(de_conv3)
        logit_msk_output = torch.sigmoid(logit_msk_output)
        # preds_msk_map = torch.softmax(logit_msk_output, dim=1)[:, 1, :, :]
        # preds_msk = torch.greater_equal(preds_msk_map, 0.5).to(torch.int32)
        # print("logit_msk_shape",logit_msk_output.shape)
        # return logit_msk_output,preds_msk_map,preds_msk
        # print("x shape: ",x.shape,"outpu shape : ",logit_msk_output.shape)
        return logit_msk_output


class very_deep_denseFCN(nn.Module):
    def __init__(self, bn_in,return_middle_map = False):
        super(very_deep_denseFCN, self).__init__()
        self.return_middle_map = return_middle_map
        self.dense_block1 = dense_block(in_channels=3, num_conv=4, kernel_size=3, filters=8, output_channels=16,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)


        self.dense_block2 = dense_block(in_channels=16, num_conv=4, kernel_size=3, filters=16, output_channels=32,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        self.dense_block3 = dense_block(in_channels=32, num_conv=4, kernel_size=3, filters=32, output_channels=64,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)

        self.dense_block4 = dense_block(in_channels=64, num_conv=4, kernel_size=3, filters=64, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.dense_block5 = dense_block(in_channels=96, num_conv=4, kernel_size=3, filters=96, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)

        self.dense_block6 = dense_block(in_channels=96, num_conv=4, kernel_size=3, filters=128, output_channels=256,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)

        self.dense_block7 = dense_block(in_channels=256, num_conv=4, kernel_size=3, filters=160, output_channels=512,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)

        self.dense_block8 = dense_block(in_channels=512, num_conv=4, kernel_size=3, filters=256, output_channels=1024,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.dense_block9 = dense_block(in_channels=1024, num_conv=4, kernel_size=3, filters=32, output_channels=16,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)


        self.final_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        processed_image = x
        spatial_input = processed_image
        '''spatial branch'''
        spatial_dense_block1 = self.dense_block1(spatial_input)  # 空域第一个dense block的输出
        spatial_dense_block2 = self.dense_block2(spatial_dense_block1)
        spatial_dense_block3 = self.dense_block3(spatial_dense_block2)
        spatial_dense_block4 = self.dense_block4(spatial_dense_block3)
        spatial_dense_block5 = self.dense_block5(spatial_dense_block4)
        spatial_dense_block6 = self.dense_block6(spatial_dense_block5)
        spatial_dense_block7 = self.dense_block7(spatial_dense_block6)
        spatial_dense_block8 = self.dense_block8(spatial_dense_block7)
        spatial_dense_block9 = self.dense_block9(spatial_dense_block8)

        output = torch.nn.functional.interpolate(spatial_dense_block9,size=(spatial_input.shape[2], spatial_input.shape[3]))

        logit_msk_output = self.final_conv(output)
        logit_msk_output = torch.sigmoid(logit_msk_output)

        if(self.return_middle_map):
            return spatial_dense_block1,spatial_dense_block2,spatial_dense_block3,spatial_dense_block4,spatial_dense_block5,\
                   spatial_dense_block6,spatial_dense_block7,de_conv1,de_conv2,de_conv3,logit_msk_output
        else:
            return logit_msk_output
