import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from random import random
from os.path import join, dirname
from data.dataset_utils import *
import pandas as pd
import os
Image.LOAD_TRUNCATED_IMAGES = True
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_retinopathy_dataset_info(req_set = 'train'):
    fgadr_root_path = '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set'
    fgadr_csv_file_name = 'DR_Seg_Grading_Label.csv'
    fgadr_image_path = '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/Original_Images'

    data_file = pd.read_csv(os.path.join(fgadr_root_path, fgadr_csv_file_name), header=None)

    db_len = 0
    start = 0
    end = 0
    # 0.7 train, 0.15 val, 0.15 test
    if req_set == 'train':
        start = 0
        end = 1289
    elif req_set == 'val':
        start = 1289
        end = 1565
    elif req_set == 'test':
        # start = 1565
        start = 1289
        end = len(data_file) - 1

    name_list = []
    label_list = []

    for i in range(start, end):
        if os.path.isfile(os.path.join(fgadr_image_path, data_file.iloc[i][0])):
            name_list.append(os.path.join(fgadr_image_path, data_file.iloc[i][0]))
            label_list.append(data_file.iloc[i][1])
        else:
            print()

    # for idx, row in data_file.iterrows():
    #     if os.path.isfile(os.path.join(fgadr_image_path, row[0])):
    #         name_list.append(os.path.join(fgadr_image_path, row[0]))
    #         label_list.append(row[1])
    #     else:
    #         print()

    return name_list, label_list


class FGADRDataset(data.Dataset):
    def __init__(self, name, split='train', val_size=0, rot_classes=3,
            img_transformer=None, bias_whole_image=None, mode='RGB'):
        # if split == 'train':
        #     names, _, labels, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % name), val_size)
        # elif split =='val':
        #     _, names, _, labels = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % name), val_size)
        # elif split == 'test':
        #     names, labels = get_dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % name))


        names, labels = get_retinopathy_dataset_info(req_set=split)


        # print(names, labels)
        # self.data_path = join(dirname(__file__), '..', 'datasets')
        self.data_path = '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/Original_Images'
        self.names = names
        self.labels = labels
        self.rot_classes = rot_classes
        self.mode = mode

        self.N = len(self.names)
        self.bias_whole_image = bias_whole_image
        self._image_transformer = img_transformer

    def rotate_all(self, img):
        """Rotate for all angles"""
        img_rts = []
        for lb in range(self.rot_classes + 1):
            img_rt = self.rotate(img, rot=lb * 90)
            img_rts.append(img_rt)

        return img_rts

    def rotate(self, img, rot):
        if rot == 0:
            img_rt = img
        elif rot == 90:
            img_rt = img.transpose(Image.ROTATE_90)
        elif rot == 180:
            img_rt = img.transpose(Image.ROTATE_180)
        elif rot == 270:
            img_rt = img.transpose(Image.ROTATE_270)
        else:
            raise ValueError('Rotation angles should be in [0, 90, 180, 270]')
        return img_rt

    def get_image(self, index):
        # framename = self.data_path + '/' + self.names[index]
        framename = self.names[index]
        img = Image.open(framename).convert(self.mode)
        img = img.resize((222, 222), Image.NEAREST)
        return img

    def __getitem__(self, index):
        img = self.get_image(index)
        rot_imgs = self.rotate_all(img)

        order = np.random.randint(self.rot_classes + 1)  # added 1 for class 0: unrotated
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0

        data = rot_imgs[order]
        data = self._image_transformer(data)
        sample = {'images': data,
                'images_ori': self._image_transformer(rot_imgs[0]),
                'aux_labels': int(order),
                'class_labels': int(self.labels[index])}
        return sample

    def __len__(self):
        return len(self.names)

class FGADRTestDataset(FGADRDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        # framename = self.data_path + '/' + self.names[index]
        framename = self.names[index]
        img = Image.open(framename).convert(self.mode)

        sample = {'images': self._image_transformer(img),
                'aux_labels': 0,
                'class_labels': int(self.labels[index])}
        return sample
