import torch
import torch.nn as nn

import numpy as np

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
        self.de_conv2 = ConvX(in_channels=64, filters=48, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)
        self.dense_block7 = dense_block(in_channels=64, num_conv=2, kernel_size=3, filters=32, output_channels=48,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)


        self.de_conv3 = ConvX(in_channels=48, filters=32, kernel_size=5, strides=1, padding=2, weight_decay=0,
                              bn_in=bn_in, dilate_rate=1, is_training=True)

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

        if(self.return_middle_map):
            return spatial_dense_block5,logit_msk_output
        else:
            return logit_msk_output
