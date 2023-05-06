import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import math
import torchvision.models as models
import torch.optim as optim
import torch.utils.data as data
import pdb
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import numpy as np
import scipy.io
import time
import torch.nn.functional as F
# from scipy.misc import imresize
import cv2
from scipy import misc
import datetime
resize_width = 384
resize_height = 384
data_txt_addr = './data/BSDS/img_train_pair_rotated.txt'
sample_num = 20000

def rand_sampler():
    """ generate a text file subsample dataset"""
    with open(data_txt_addr) as f:
        lines = f.readlines()

    ind = np.random.choice(len(lines), sample_num)
    new_lines = []
    for i in range(len(ind)):
        new_lines.append(lines[ind[i]])

    f = open('./data/BSDS/bsds1.txt', 'w')
    f.write("".join(new_lines))
    f.close()

def default_loader(path, data_name):
    if data_name == 'imlist':
        return Image.open(path)  # misc.imread(path)#Image.open(path).convert('RGB')
    else:
        mat = scipy.io.loadmat(path)
        return np.asarray(mat['edge_gt'])  # Image.open(path).convert('L')


def default_flist_reader(root, flist, data_name):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    labellist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, labelpath = line.strip().split()
            impath = root + '/' + impath
            labelpath = root + '/' + labelpath
            # pdb.set_trace()
            imlist.append(impath)
            labellist.append(labelpath)

    return imlist, labellist

def lable_flist_reader(root, flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    labellist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            labelpath_ = line.strip().split()[0]
            #print(labelpath_)
            labelpath = root + '/' + labelpath_
            #print(labelpath)
            labellist.append(labelpath)

    return labellist

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, data_name, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        flist = root + '/' + flist
        # flist = flist
        assert data_name == 'imlist' or data_name == 'labellist'
        self.imlist, self.labellist = flist_reader(root, flist, data_name)

        # pdb.set_trace()
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        labelpath = self.labellist[index]
        img = self.loader(os.path.join(self.root, impath), data_name='imlist')

        img = transforms.ToTensor()(img)
        img = img[:, 1:img.size(1), 1:img.size(2)]
        img = img.float()

        label = self.loader(os.path.join(self.root, labelpath), data_name='labellist')
        label = torch.from_numpy(label)
        label = label[1:label.size(0), 1:label.size(1)]
        label = label.unsqueeze(0)
        label = label.float()

        if self.transform is not None:
            img = self.transform(img)
            # label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.imlist)


class TestFilelist(data.Dataset):
    def __init__(self, root, flist, data_name, transform=None, target_transform=None,
                 flist_reader=lable_flist_reader, loader=default_loader):
        self.root = root
        flist = root + '/' + flist
        # flist = flist
        assert data_name == 'imlist' or data_name == 'labellist'
        self.labellist = flist_reader(root, flist)

        # pdb.set_trace()
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        return len(self.labellist)

    def __getitem__(self, index):

        labelpath = self.labellist[index]

        label = self.loader(os.path.join(self.root, labelpath), data_name='imlist')
        label = transforms.ToTensor()(label)
        label = label[:, 1:label.size(1), 1:label.size(2)]
        label = label.float()



        return label