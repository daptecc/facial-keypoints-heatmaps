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
        
class CPM2(nn.Module):
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
        if stage == 1:
            out = self.block1a(input)
        elif stage == 2:
            out = self.block1b(input)
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

class CPM(nn.Module):
    def __init__(self, n_keypoints, channels=1):
        super(CPM, self).__init__()
        self.k = n_keypoints
        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        self.conv1_stage1 = nn.Conv2d(channels, 128, kernel_size=9, padding=4)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_stage1 = nn.Conv2d(512, self.k + 1, kernel_size=1)

        self.conv1_stage2 = nn.Conv2d(channels, 128, kernel_size=9, padding=4)
        self.pool1_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        
        self.Mconv1_stage2 = nn.Conv2d(32 + (self.k + 1)*2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage2 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage3 = nn.Conv2d(32 + (self.k + 1)*2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage3 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage4 = nn.Conv2d(32 + (self.k + 1)*2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage4 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage5 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage5 = nn.Conv2d(32 + (self.k + 1)*2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage5 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage6 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage6 = nn.Conv2d(32 + (self.k + 1)*2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage6 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

    def _stage1(self, image):

        x = self.pool1_stage1(F.relu(self.conv1_stage1(image)))
        x = self.pool2_stage1(F.relu(self.conv2_stage1(x)))
        x = self.pool3_stage1(F.relu(self.conv3_stage1(x)))
        x = F.relu(self.conv4_stage1(x))
        x = F.relu(self.conv5_stage1(x))
        x = F.relu(self.conv6_stage1(x))
        x = self.conv7_stage1(x)

        return x

    def _middle(self, image):

        x = self.pool1_stage2(F.relu(self.conv1_stage2(image)))
        x = self.pool2_stage2(F.relu(self.conv2_stage2(x)))
        x = self.pool3_stage2(F.relu(self.conv3_stage2(x)))

        return x

    def _stage2(self, pool3_stage2_map, conv7_stage1_map, pool_center_map):

        x = F.relu(self.conv4_stage2(pool3_stage2_map))
        x = torch.cat([x, conv7_stage1_map, pool_center_map], dim=1)
        x = F.relu(self.Mconv1_stage2(x))
        x = F.relu(self.Mconv2_stage2(x))
        x = F.relu(self.Mconv3_stage2(x))
        x = F.relu(self.Mconv4_stage2(x))
        x = self.Mconv5_stage2(x)

        return x

    def _stage3(self, pool3_stage2_map, Mconv5_stage2_map, pool_center_map):

        x = F.relu(self.conv1_stage3(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage2_map, pool_center_map], dim=1)
        x = F.relu(self.Mconv1_stage3(x))
        x = F.relu(self.Mconv2_stage3(x))
        x = F.relu(self.Mconv3_stage3(x))
        x = F.relu(self.Mconv4_stage3(x))
        x = self.Mconv5_stage3(x)

        return x

    def _stage4(self, pool3_stage2_map, Mconv5_stage3_map, pool_center_map):

        x = F.relu(self.conv1_stage4(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage3_map, pool_center_map], dim=1)
        x = F.relu(self.Mconv1_stage4(x))
        x = F.relu(self.Mconv2_stage4(x))
        x = F.relu(self.Mconv3_stage4(x))
        x = F.relu(self.Mconv4_stage4(x))
        x = self.Mconv5_stage4(x)

        return x

    def _stage5(self, pool3_stage2_map, Mconv5_stage4_map, pool_center_map):

        x = F.relu(self.conv1_stage5(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage4_map, pool_center_map], dim=1)
        x = F.relu(self.Mconv1_stage5(x))
        x = F.relu(self.Mconv2_stage5(x))
        x = F.relu(self.Mconv3_stage5(x))
        x = F.relu(self.Mconv4_stage5(x))
        x = self.Mconv5_stage5(x)

        return x

    def _stage6(self, pool3_stage2_map, Mconv5_stage5_map, pool_center_map):
        
        x = F.relu(self.conv1_stage6(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage5_map, pool_center_map], dim=1)
        x = F.relu(self.Mconv1_stage6(x))
        x = F.relu(self.Mconv2_stage6(x))
        x = F.relu(self.Mconv3_stage6(x))
        x = F.relu(self.Mconv4_stage6(x))
        x = self.Mconv5_stage6(x)

        return x

    
    def forward(self, image, center_map):

        pool_center_map = self.pool_center(center_map)

        conv7_stage1_map = self._stage1(image)

        pool3_stage2_map = self._middle(image)

        Mconv5_stage2_map = self._stage2(pool3_stage2_map, conv7_stage1_map,
                                         pool_center_map)
        Mconv5_stage3_map = self._stage3(pool3_stage2_map, Mconv5_stage2_map,
                                         pool_center_map)
        Mconv5_stage4_map = self._stage4(pool3_stage2_map, Mconv5_stage3_map,
                                         pool_center_map)
        Mconv5_stage5_map = self._stage5(pool3_stage2_map, Mconv5_stage4_map,
                                         pool_center_map)
        Mconv5_stage6_map = self._stage6(pool3_stage2_map, Mconv5_stage5_map,
                                         pool_center_map)

        return [conv7_stage1_map, Mconv5_stage2_map, Mconv5_stage3_map, Mconv5_stage4_map, Mconv5_stage5_map, Mconv5_stage6_map]
