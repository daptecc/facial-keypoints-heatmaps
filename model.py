import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvBlock(nn.Module):
    def __init__(self, nconvs, in_channel, out_channel):
        super().__init__()
        layers = []
        for i in range(nconvs):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            in_channel = out_channel
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, input):
        out = self.conv_block(input)
        return out
        
class CPM(nn.Module):
    def __init__(self, n_keypoints, channels=1):
        super().__init__()
        self.k = n_keypoints
        self.channels = channels
        self.block1a = ConvBlock(nconvs=2, in_channel=channels, out_channel=64)
        self.block1b = ConvBlock(nconvs=2, in_channel=channels + self.k, out_channel=64)
        self.block2 = ConvBlock(nconvs=2, in_channel=64, out_channel=128)
        self.block3 = ConvBlock(nconvs=3, in_channel=128, out_channel=256)
        self.block4 = ConvBlock(nconvs=3, in_channel=256, out_channel=512)
        self.block5 = ConvBlock(nconvs=3, in_channel=512, out_channel=512)

        self.conv6 = nn.Conv2d(512, 4096, kernel_size=1, stride=1, padding=0)
        self.conv7 = nn.Conv2d(4096, 15, kernel_size=1, stride=1, padding=0)
        
        self.pool3 = nn.Conv2d(256, 15, kernel_size=1, stride=1, padding=0)
        self.pool4 = nn.Conv2d(512, 15, kernel_size=1, stride=1, padding=0)
        
        self.up_pool4 = nn.ConvTranspose2d(15, 15, kernel_size=2, stride=2)
        self.up_conv7 = nn.ConvTranspose2d(15, 15, kernel_size=4, stride=4)
        self.up_fused = nn.ConvTranspose2d(15, 15, kernel_size=8, stride=8)
        self.up_final = nn.Conv2d(15, self.k, kernel_size=1, stride=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, input, stage):
        heatmaps = self.stage(input, stage)
        return heatmaps
        
    def stage(self, input, stage):
        #print('input', input.shape)
        out = self.block1a(input) if stage == 1 else self.block1b(input)
        #print('block 1', out.shape)
        out = self.block2(out)
        #print('block 2', out.shape)
        pool3 = self.block3(out)
        #print('pool3', pool3.shape)
        pool4 = self.block4(pool3)
        #print('pool4', pool4.shape)
        out = self.block5(pool4)
        #print('block 5', out.shape)
        
        out = self.conv6(out)
        #print('conv6', out.shape)
        out = self.relu(out)
        out = self.conv7(out)
        #print('conv7', out.shape)
        out = self.relu(out)
        
        # upsampling
        preds_pool3 = self.pool3(pool3)
        #print('preds_pool3', preds_pool3.shape)
        preds_pool3 = self.relu(preds_pool3)
        preds_pool4 = self.pool4(pool4)
        #print('preds_pool4', preds_pool4.shape)
        preds_pool4 = self.relu(preds_pool4)
        up_pool4 = self.up_pool4(preds_pool4)
        #print('up_pool4', preds_pool3.shape)
        up_pool4 = self.relu(up_pool4)
        up_conv7 = self.up_conv7(out)
        #print('up_conv7', preds_pool3.shape)
        up_conv7 = self.relu(up_conv7)
        
        fused = torch.add(preds_pool3, up_pool4)
        fused = torch.add(fused, up_conv7)
        #print('fused', fused.shape)
        
        heatmaps = self.up_fused(fused)
        #print('heatmaps transpose', heatmaps.shape)
        heatmaps = self.relu(heatmaps)
        heatmaps = self.up_final(heatmaps)
        #print('heatmaps conv2d', heatmaps.shape)
        
        return heatmaps
