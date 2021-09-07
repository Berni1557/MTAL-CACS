#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 09:41:53 2021

@author: Bernhard Foellmer
"""

import torch
from torch import nn, optim

class MTALModel():
    """
    MTALModel - Multi task model
    """
    
    def __init__(self, device='cuda'):
        
        # Init params
        self.params=dict()
        self.params['lr'] = 0.005
        self.params['device'] = device

    def create(self):
        """
        Create model
        """
        class Conv_down(nn.Module):
            def __init__(self, in_ch, out_ch):
                super(Conv_down, self).__init__()
                self.down = nn.Conv2d(in_ch, out_ch,  kernel_size=4, stride=2, padding=1)
                self.relu1 = nn.LeakyReLU(0.2)
                self.dropout = nn.Dropout(p=0.0)
                self.conv = nn.Conv2d(out_ch, out_ch,  kernel_size=3, stride=1, padding=1)
                self.norm = nn.BatchNorm2d(out_ch)
                self.relu2 = nn.LeakyReLU(0.2)
                self.down.weight.data.normal_(0.0, 0.1)
                self.conv.weight.data.normal_(0.0, 0.1)
        
            def forward(self, x):
                x = self.down(x)
                x = self.relu1(x)
                x = self.dropout(x)
                x = self.conv(x)
                x = self.norm(x)
                x = self.relu2(x)
                return x

        class Conv_up(nn.Module):
            def __init__(self, in_ch, out_ch, kernel_size_1=3, stride_1=1, padding_1=1, kernel_size_2=3, stride_2=1, padding_2=1):
                super(Conv_up, self).__init__()
                self.up = nn.UpsamplingBilinear2d(scale_factor=2)
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size_1, padding=padding_1, stride=stride_1)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size_2, padding=padding_2, stride=stride_2)
                self.relu1 = nn.LeakyReLU(0.2)
                self.relu2 = nn.LeakyReLU(0.2)
                self.dropout = nn.Dropout(p=0.0)
                self.norm = nn.BatchNorm2d(out_ch)
                self.conv1.weight.data.normal_(0.0, 0.1)
                self.conv2.weight.data.normal_(0.0, 0.1)
        
            def forward(self, x1, x2):
                x1 = self.up(x1)
                x = torch.cat((x1, x2), dim=1)
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.dropout(x)
                x = self.conv2(x)
                x = self.norm(x)
                x = self.relu2(x)
                return x
            
        class MTAL(nn.Module):
            def __init__(self):
                super(MTAL, self).__init__()

                #self.conv_down1 = Conv_down(props['Input_size'][2], 64)
                
                self.conv_down1 = Conv_down(16, 16)
                self.conv_down2 = Conv_down(16, 32)
                self.conv_down3 = Conv_down(32, 32)
                self.conv_down4 = Conv_down(32, 64)
                self.conv_down5 = Conv_down(64, 64)
                self.conv_down6 = Conv_down(64, 64)
                self.conv_down7 = Conv_down(64, 128)
                self.conv_down8 = Conv_down(128, 128)
                
                self.conv_up1 = Conv_up(128+128, 128)
                self.conv_up2 = Conv_up(128+64, 64)
                self.conv_up3 = Conv_up(64+64, 64)
                self.conv_up4 = Conv_up(64+64, 64)
                self.conv_up5 = Conv_up(64+32, 32)
                self.conv_up6 = Conv_up(32+32, 32)
                self.conv_up7 = Conv_up(32+16, 16)
                self.conv_up8 = Conv_up(16+16, 16)

                self.conv_up1_class = Conv_up(128+128+128, 128)
                self.conv_up2_class = Conv_up(128+64+64, 64)
                self.conv_up3_class = Conv_up(64+64+64, 64)
                self.conv_up4_class = Conv_up(64+64+64, 64)
                self.conv_up5_class  = Conv_up(64+32+32, 32)
                self.conv_up6_class  = Conv_up(32+32+32, 32)
                self.conv_up7_class  = Conv_up(32+16+16, 16)
                self.conv_up8_class  = Conv_up(16+16+16+2, 32)
                # self.conv_up6_class  = Conv_up(32+32, 32)
                # self.conv_up7_class  = Conv_up(32+16, 16)
                # self.conv_up8_class  = Conv_up(16+16+2, 32)
                
                self.conv00 = nn.Conv2d(2, 16, kernel_size=5, padding=2, stride=1)
                self.relu00 = nn.LeakyReLU(0.2)
                self.conv01 = nn.Conv2d(16, 16, kernel_size=5, padding=2, stride=1)
                self.relu01 = nn.LeakyReLU(0.2)
                
                self.conv_double_out = nn.Sequential(
                    nn.Conv2d(16, 4,  kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(4, 4,  kernel_size=3, stride=1, padding=1),
                    nn.Softmax(dim=1),
                )
                
                self.conv0 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)
                self.relu0 = nn.LeakyReLU(0.2)
                self.conv1 = nn.Conv2d(16, 2, kernel_size=3, padding=1, stride=1)
                self.soft = nn.Softmax(dim=1)

                self.dropout0 = nn.Dropout(p=0.5)
                

            def forward(self, x):

                x00 = self.conv00(x)
                x00r = self.relu00(x00)
                x01 = self.conv01(x00r)
                x01r = self.relu01(x01)
                
                x1 = self.conv_down1(x01r)
                x2 = self.conv_down2(x1)
                x3 = self.conv_down3(x2)
                x4 = self.conv_down4(x3)
                x5 = self.conv_down5(x4)
                x6 = self.conv_down6(x5)
                x7 = self.conv_down7(x6)
                x8 = self.conv_down8(x7)
                x8d = self.dropout0(x8)
                
                x9 = self.conv_up1(x8d, x7)
                x10 = self.conv_up2(x9, x6)
                x11 = self.conv_up3(x10, x5)
                x12 = self.conv_up4(x11, x4)
                x13 = self.conv_up5(x12, x3)
                x14 = self.conv_up6(x13, x2)
                x15 = self.conv_up7(x14, x1)
                x16 = self.conv_up8(x15, x01r)
                xout = self.conv_double_out(x16)

                x9c = self.conv_up1_class(x8d, torch.cat((x9, x7), dim=1))
                x10c = self.conv_up2_class(x9c, torch.cat((x10, x6), dim=1))

                x11c = self.conv_up3_class(x10c, torch.cat((x11, x5), dim=1))
                x12c = self.conv_up4_class(x11c, torch.cat((x12, x4), dim=1))
                x13c = self.conv_up5_class(x12c, torch.cat((x13, x3), dim=1))
                x14c = self.conv_up6_class(x13c, torch.cat((x14, x2), dim=1))
                x15c = self.conv_up7_class(x14c, torch.cat((x15, x1), dim=1))
                x16c = self.conv_up8_class(x15c, torch.cat((x16, x01r, x), dim=1))

                xoutc1 = self.conv0(x16c)
                xoutc2 = self.relu0(xoutc1)
                xoutc3 = self.conv1(xoutc2)
                xoutc4 = self.soft(xoutc3)

                return xout, xoutc4

        # Create model
        mtal = MTAL()
        mtal.train()
        if self.params['device']=='cuda':
            mtal.cuda() 
        self.mtal=mtal

        # Set default learning rate
        self.opt_unet_prior = optim.Adam(self.mtal.parameters(), lr = self.params['lr'], betas=(0.9, 0.999), weight_decay=0.01)

    def load(self, modelpath):
        """
        Load pretained model
        """
        self.mtal.load_state_dict(torch.load(modelpath))
        
    def predict(self, Xin):
        self.mtal.eval()
        with torch.no_grad():
            Y_region, Y_lesion = self.mtal(Xin)
        return Y_region, Y_lesion

