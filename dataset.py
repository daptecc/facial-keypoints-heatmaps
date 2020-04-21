import os, glob, math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FacialKeyPointsDataset(Dataset):
    
    def __init__(self, csv_file, n_keypoints, size, transform):
        """
        Args:
            csv_file (string): csv file with facial keypoints and pixel values for all images
            n_keypoints (int): number of facial keypoints (w, h)
            size (int,int): shape of image
            transform: torchvision transforms
        """
        
        self.df = self.filter_by_n_kpts(csv_file, n_keypoints)
        self.kp_mean = self.df.iloc[:,:30].stack().mean()
        self.kp_std = self.df.iloc[:,:30].stack().std()
        self.n_keypoints = n_keypoints
        self.size = size
        self.transform = transform


    def filter_by_n_kpts(self, csv_file, n_keypoints):
        
        df = pd.read_csv(csv_file)
        notnan = df.apply(lambda x: np.sum(~x[:30].isnull()), axis=1)
        subdf = df[notnan == n_keypoints * 2]
        subdf = subdf.reset_index(drop=True)
        
        return subdf
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        img = np.array(self.df.iloc[idx, 30].split())
        w, h = self.size
        img = img.astype(np.float32).reshape(w, h)
        #img = np.expand_dims(img, axis=0)

        keypts = self.df.iloc[idx,:30].values
        keypts = keypts.reshape(-1, 1)
        keypts = keypts.astype('float')
        
        if self.n_keypoints == 4:
            keypts = keypts[~np.isnan(keypts)]
        
        heatmaps = np.zeros((h, w, len(keypts)//2), dtype=np.float32)
        for i in range(0, len(keypts)//2):
            x = int(keypts[i * 2])
            y = int(keypts[i * 2 + 1])
            heatmap = self.gaussian(x, y, w, h)
            heatmaps[:,:, i] = heatmap
                    
        #heatmaps[:,:, 0] = 1.0 - np.max(heatmaps, axis=2)
        
        if self.transform:
            img = self.transform(img)
            heatmaps = self.transform(heatmaps)
        
        return img, heatmaps
    
    
    def gaussian(self, x, y, H, W, sigma=5):
        """
        Create heatmaps by convoluting a 2D gaussian kernel over a (x,y) keypoint
        """
        channel = [math.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))

        return channel
    
    
    def show_sample(self, img, heatmap):
        plt.imshow(img.reshape(self.size), cmap='gray', alpha=0.5)
        plt.imshow(heatmap, alpha=0.5)
    
    def display_sample(self):
        fig = plt.figure(figsize=(20, 6))
        fig.tight_layout()
        rand_i = np.random.randint(0, len(self.df))
        img, heatmaps = self.__getitem__(rand_i)
        n = len(heatmaps)
        #fig.add_subplot(2, (n + 1)//2, 1)
        #plt.imshow(img.reshape(self.size), cmap='gray')

        for i in range(0, n):
            heatmap = heatmaps[i]
            fig.add_subplot(2, (n + 1)//2, i + 1)
            self.show_sample(img, heatmap)
        fig.savefig('samples/heatmaps.png')
