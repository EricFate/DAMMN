# -*- coding: utf-8 -*-
import torch
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as Data
import torchvision.transforms as transforms
import os
from config import opt

class BaseLineDataset(Data.Dataset):
    def __init__(self, f):
        super(BaseLineDataset, self).__init__()
        self.train_set_x, self.train_set_y, self.unlabel_x, self.unlabel_y, self.test_set_x, self.test_set_y = rearrange_data(f)


    def __getitem__(self, index):
        if opt.train:
            return self.get_item(self.train_set_x,self.train_set_y,index)
        else:
            return self.get_item(self.test_set_x,self.test_set_y,index)

    def get_item(self, x, y, index):
        if opt.generation_method == 'none':
            return x[index], y[index]
        else:
            if opt.generation_method == 'sep':
                modals = _seperate_image(x[index:index + 1, :, :, :])
            elif opt.generation_method == 'noise':
                modals = _add_noise(x[index:index + 1, :, :, :])
            else:
                modals = _separation_noise(x[index:index + 1, :, :, :])
            return modals[opt.baseline_modal][0], y[index]

    def __len__(self):
        if opt.train:
            return self.train_set_x.size(0)
        else:
            return self.test_set_x.size(0)

class SemiSupervisedDataset(Data.Dataset):
    def __init__(self, f):
        super(SemiSupervisedDataset, self).__init__()
        self.train_set_x, self.train_set_y, self.unlabel_x, self.unlabel_y, self.test_set_x, self.test_set_y = rearrange_data(f)
        self.prop = opt.prop
        self.data_generation_method = None

    def __getitem__(self, index):
        if opt.train:
            train_left_x, train_right_x = self.data_generation_method(
                self.train_set_x[index * self.prop[0]:(index + 1) * self.prop[0]])
            train_y = self.train_set_y[index * self.prop[0]:(index + 1) * self.prop[0]]
            unlabel_left_x, unlabel_right_x = self.data_generation_method(
                self.unlabel_x[index * self.prop[1]:(index + 1) * self.prop[1]])
            return train_left_x, train_right_x, train_y, unlabel_left_x, unlabel_right_x
        else:
            test_left_x, test_right_x = self.data_generation_method(self.test_set_x[index:index + 1, :, :, :])
            return test_left_x[0], test_right_x[0], self.test_set_y[index]

    def __len__(self):
        if opt.train:
            return int(self.train_set_x.size(0) / self.prop[0])
        else:
            return self.test_set_x.size(0)

class NoiseDataset(SemiSupervisedDataset):
    def __init__(self, f):
        super(NoiseDataset, self).__init__(f)
        self.data_generation_method = _add_noise


class SeperationDataset(SemiSupervisedDataset):
    def __init__(self, f ):
        super(SeperationDataset, self).__init__(f)
        self.data_generation_method = _seperate_image

class SeparationNoiseDataset(SeperationDataset):
    def __init__(self, f ):
        super(SeperationDataset, self).__init__(f)
        self.data_generation_method = _separation_noise


def rearrange_data(f):
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    test_set_x, test_set_y = parse_data(test_set)
    valid_set_x, valid_set_y = parse_data(valid_set)
    train_set_x, train_set_y = parse_data(train_set)
    train_set_x = torch.cat((train_set_x, valid_set_x))
    train_set_y = torch.cat((train_set_y, valid_set_y))
    n  = train_set_x.size(0)
    if opt.perm is None:
        opt.perm = np.random.permutation(n)
    train_set_x = train_set_x[opt.perm]
    train_set_y = train_set_y[opt.perm]
    n_train = int( opt.prop[0] * n / (opt.prop[0]+opt.prop[1]))
    print(n_train)
    train_x = train_set_x[:n_train]
    train_y = train_set_y[:n_train]
    unlabel_x  = train_set_x[n_train:]
    unlabel_y = train_set_y[n_train:]
    return train_x,train_y,unlabel_x,unlabel_y,test_set_x,test_set_y


def _add_noise(img):
    normal = torch.randn((1, 1, 28, 28)) * 0.03 + 0.3
    pepper = 1 - (torch.randint(0, 4, (1, 1, 28, 28)) / 3).float() / 2
    left = img + normal
    right = img * pepper
    return left, right


def _seperate_image(img):
    left = img.clone()
    right = img.clone()
    left[:, :, :opt.sep, :opt.sep] = 0
    right[:, :, opt.sep:, -opt.sep:] = 0
    return left, right

def _separation_noise(img):
    normal = torch.randn((1, 1, 28, 28)) * 0.03 + 0.3
    pepper = 1 - (torch.randint(0, 4, (1, 1, 28, 28)) / 3).float() / 2
    left = img + normal
    right = img * pepper
    left[:, :, :opt.sep, :opt.sep] = 0
    right[:, :, opt.sep:, -opt.sep:] = 0
    return left, right

def parse_data(data_xy):
    data_x, data_y = data_xy
    data_x = torch.from_numpy(np.asarray(data_x, dtype=np.float32))
    data_x = data_x.view(data_x.size(0), 1, 28, 28)
    data_y = torch.from_numpy(np.asarray(data_y, dtype=np.int64))
    return data_x, data_y
