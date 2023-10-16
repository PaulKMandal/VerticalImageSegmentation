# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from federatedml.nn.dataset.segmentation import image_transforms
from PIL import Image
from torchvision.transforms import functional as F
import numpy.lib.recfunctions as nlr

root_dir = "CamVid/"
data_dir = os.path.join(root_dir, "train")
label_dir = os.path.join(root_dir, "train_labels")
label_colors_file = os.path.join(root_dir, "class_dict.csv")
val_data_dir = os.path.join(root_dir, "val")
val_label_dir = os.path.join(root_dir, "val_labels")

num_class = 32
means = np.array([103.939, 116.779, 123.68]) / 255.  # mean of three channels in the order of BGR
h,w = 128,128


val_h = int(h / 32) * 32  # 704
val_w = w  # 960
class_to_regress = 18 #road


class SegmentationLabel(Dataset):
    def __init__(self, seg_dir, map_file, transform=image_transforms.LblToTensor()):
        from glob import glob
        from os import path
        self.seg_dir = seg_dir
        self.files = []
        self.crop = True
        self.color_to_class = {}
        self.class_to_color = {}
        self.n_class = 32
        #self.h = 512
        #self.w = 512
        self.h = 128
        self.w = 128
        
        with open(map_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row_num, row in enumerate(reader, start=1):
                color = (int(row['r']), int(row['g']), int(row['b']))
                # class_name = row['name']
                self.color_to_class[color] = row_num - 1  # class_name

            self.n_class = len(self.color_to_class)

        self.class_to_color = {v: k for k, v in self.color_to_class.items()}
        
        for im_f in glob(path.join(seg_dir, '*.png')):
            self.files.append(os.path.basename(im_f).replace('_L.png', ''))
            # self.files.append(im_f.replace('_im.jpg', ''))

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]

        lbl = Image.open(self.seg_dir + b + '_L.png')

        lbl = lbl.resize((self.h,self.w))


        
        if self.transform is not None:
            lbl = self.transform(lbl)

        # print("lbl0",lbl.size())

        lbl = lbl.permute(1, 2, 0)
        lbl = lbl.numpy()
        lbl = nlr.unstructured_to_structured(lbl).astype('O')
        dummy_class = 31 #Change dummy class to whatever class you want colors not defined in your dictinoary to be
        lbl = torch.tensor(np.vectorize(self.color_to_class.get)(lbl, dummy_class), dtype=torch.uint8)

        # create one-hot encoding

        target = torch.nn.functional.one_hot(lbl.to(torch.int64), num_classes=self.n_class)
        target = target.permute(2, 0, 1)
        target = target.float()
        
        return target[17] #.unsqueeze(0)

class SegmentationImage(Dataset):
    def __init__(self, img_dir, transform=image_transforms.ImgToTensor()):
        from glob import glob
        from os import path
        self.img_dir = img_dir
        self.files = []
        self.crop = True
        self.color_to_class = {}
        self.class_to_color = {}
        self.n_class = 32
        self.h = 128
        self.w = 128
        
        for im_f in glob(path.join(img_dir, '*.png')):
            self.files.append(os.path.basename(im_f).replace('.png', ''))
            # self.files.append(im_f.replace('_im.jpg', ''))

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        img = Image.open(self.img_dir + b + '.png')

        img = img.resize((self.h,self.w))

        if self.transform is not None:
            img = self.transform(img)
        
        return img

def show_batch(batch):
    img_batch = batch['X']
    img_batch[:, 0, ...].add_(means[0])
    img_batch[:, 1, ...].add_(means[1])
    img_batch[:, 2, ...].add_(means[2])
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))

    plt.title('Batch from dataloader')


def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))
