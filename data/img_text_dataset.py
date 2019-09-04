# -*- coding: utf-8 -*-
import torch
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as Data
import torchvision.transforms as transforms
import os
from config import opt

# prefix = "G:\\data\\coco\\"
# prefix = "/data/yangy/coco/"
# prefix = "/data/yangy/COCO/"
prefix = "/home/amax/data/"


param_map = {"MS-COCO": {"image_record": prefix+"COCO/select_20coco_imgs.pkl",
                         # "image_dir": prefix+"COCO/images/train/",
                         "image_dir": prefix+"COCO/train2014/",
                         # "image_dir": prefix+"images\\",
                         "text_file": prefix+"COCO/select_20coco_text.npy",
                         "label_file": prefix+"COCO/select_20coco_label.npy"},
             "NUS-WIDE": {"image_record": prefix+"NUS-WIDE/imgs.pkl",
                          "image_dir": prefix+"NUS-WIDE/images/",
                          "text_file": prefix+"NUS-WIDE/text.npy",
                          "label_file": prefix+"NUS-WIDE/label.npy"},
             "FLICKR": {"image_dir": prefix+"mirflickr/picture/",
                        "text_file": prefix+"mirflickr/text_data.npy",
                        "label_file": prefix+"mirflickr/label_data.npy"},
             "IAPR": {"image_dir": prefix+"iapr/images/",
                      "text_dir": prefix+"iapr/text/",
                      "label_file": prefix+"iapr/label.npy",
                      "label_to_image":prefix+"iapr/labelidTOimgid.npy"}
             }

# param_map = {"MS-COCO": {"image_record": "G:\\data\\coco\\select_20coco_imgs.pkl",
#                          "image_dir": "G:\\data\\coco\\images\\",
#                          "text_file": "G:\\data\\coco\\select_20coco_text.npy",
#                          "label_file": "G:\\data\\coco\\select_20coco_label.npy"},
#              "NUS-WIDE": {"image_record": "G:\\data\\NUS-WIDE\\imgs.pkl",
#                           "image_dir": "G:\\data\\NUS-WIDE\\images\\",
#                           "text_file": "G:\\data\\NUS-WIDE\\text.npy",
#                           "label_file": "G:\\data\\NUS-WIDE\\label.npy"},
#              "FLICKR": {"image_dir": "G:\\data\\mirflickr\\",
#                         "text_file": "G:\\data\\mirflickr\\text_data.npy",
#                         "label_file": "G:\\data\\mirflickr\\label_data.npy"},
#              "IAPR": {"image_dir": "G:\\data\\iapr\\images\\",
#                       "text_dir": "G:\\data\\iapr\\text\\",
#                       "label_file": "G:\\data\\iapr\\label.npy",
#                       "label_to_image":"G:\\data\\iapr\\labelidTOimgid.npy"}
#              }

dataset_names = ["MS-COCO"
    , "NUS-WIDE", "FLICKR", "IAPR"
                 ]


class MyDataset(Data.Dataset):
    def __init__(self, traintestproportion=0.667, superviseunsuperviseproportion=(3, 6),
                 dataset_name="MS-COCO",pre_set = False):
        super(MyDataset, self).__init__()
        if dataset_name not in dataset_names:
            raise Exception("illegal dataset name: " + dataset_name)
        # self.imgfilenamerecord = imgfilenamerecord
        # self.imgfilename = img_file_dir
        # self.textfilename = textfilename
        # self.labelfilename = labelfilename
        self.pro1 = traintestproportion
        self.pro2 = superviseunsuperviseproportion

        # print(self.pro2)
        # exit(-1)

        # print(self.labellist.shape)
        # exit(-1)
        data_param = param_map[dataset_name]
        if dataset_name == "MS-COCO" or dataset_name == "NUS-WIDE":
            fr = open(data_param["image_record"], 'rb')
            self.imgrecordlist = pickle.load(fr)
            for i in range(len(self.imgrecordlist)):
                self.imgrecordlist[i] = data_param["image_dir"] + self.imgrecordlist[i]
            self.imgrecordlist = np.array(self.imgrecordlist)
            self.textlist = np.load(data_param["text_file"])
            self.labellist = np.load(data_param["label_file"])
        elif dataset_name == "FLICKR":
            self.imgrecordlist = np.array([data_param["image_dir"] + 'im' + str(i) + '.jpg' for i in range(1, 25001)])
            self.textlist = np.load(data_param["text_file"])
            self.labellist = np.load(data_param["label_file"])
        elif dataset_name == "IAPR":
            self.labellist = np.load(data_param["label_file"])
            label_to_image = np.load(data_param["label_to_image"])
            self.imgrecordlist = np.array([data_param["image_dir"] + str(i) + '.jpg' for i in label_to_image])
            self.textlist = np.array([np.load(data_param["text_dir"] + str(i) + '.npy') for i in label_to_image])

        '''
        upset data.
        '''
        if pre_set:
            permutation = np.load('checkpoint/permutation.npy')
        else:
            permutation = np.random.permutation(len(self.imgrecordlist))
            np.save('checkpoint/permutation.npy',permutation)

        self.imgrecordlist = self.imgrecordlist[permutation]
        self.textlist = self.textlist[permutation, :]
        self.labellist = self.labellist[permutation, :]

        self.text_input_size = self.textlist.shape[1]
        self.label_size =  self.labellist.shape[1]

        # print(int(self.pro1*self.pro2*len(self.imgrecordlist)))
        # print(int(self.pro1*len(self.imgrecordlist))- int(self.pro1*self.pro2*len(self.imgrecordlist)))
        # exit(-1)
        self.samplesize = int(int(self.pro1 * len(self.imgrecordlist)) / (self.pro2[0] + self.pro2[1]))
        # print(self.samplesize)
        self.superviseimgrecordlist = self.imgrecordlist[0:self.samplesize * self.pro2[0]]
        self.supervisetextlist = self.textlist[0:self.samplesize * self.pro2[0]]
        self.superviselabellist = self.labellist[0:self.samplesize * self.pro2[0]]

        self.unsuperviseimgrecordlist = self.imgrecordlist[self.samplesize * self.pro2[0]:self.samplesize * (
            self.pro2[0] + self.pro2[1])]
        self.unsupervisetextlist = self.textlist[
                                   self.samplesize * self.pro2[0]:self.samplesize * (self.pro2[0] + self.pro2[1])]
        self.unsuperviselabellist = self.labellist[
                                    self.samplesize * self.pro2[0]:self.samplesize * (self.pro2[0] + self.pro2[1])]
        # print(len(self.unsuperviseimgrecordlist))
        # exit(-1)
        self.testimgrecordlist = self.imgrecordlist[
                                 int(self.pro1 * len(self.imgrecordlist)):len(self.imgrecordlist)]
        self.testtextlist = self.textlist[int(self.pro1 * len(self.textlist)):len(self.textlist)]
        self.testlabellist = self.labellist[int(self.pro1 * len(self.labellist)):len(self.labellist)]

    def __getitem__(self, index):
        if opt.train == True and opt.supervise == True:
            img = Image.open(self.superviseimgrecordlist[index]).convert('RGB').resize((256, 256))
            text = self.supervisetextlist[index]
            label = self.superviselabellist[index]
            img = transforms.ToTensor()(img).float()
            text = torch.FloatTensor(text)
            label = torch.FloatTensor(label)
            return img,text, label
        elif opt.train == True and opt.supervise == False:
            supervise_img = []
            supervise_text = []
            supervise_label = []
            for i in range(index * self.pro2[0], (index + 1) * self.pro2[0]):
                temp_img = Image.open(self.superviseimgrecordlist[i]).convert('RGB').resize((256, 256))
                temp_text = self.supervisetextlist[i]
                temp_label = self.superviselabellist[i]
                temp_img = transforms.ToTensor()(temp_img).float()
                temp_text = torch.FloatTensor(temp_text)
                temp_label = torch.FloatTensor(temp_label)
                supervise_img.append(temp_img)
                supervise_text.append(temp_text)
                supervise_label.append(temp_label)
            unsupervise_img = []
            unsupervise_text = []
            unsupervise_label = []
            for i in range(index * self.pro2[1], (index + 1) * self.pro2[1]):
                temp_img = Image.open(self.unsuperviseimgrecordlist[i]).convert('RGB').resize((256, 256))
                temp_text = self.unsupervisetextlist[i]
                temp_label = self.unsuperviselabellist[i]
                temp_img = transforms.ToTensor()(temp_img).float()
                temp_text = torch.FloatTensor(temp_text)
                temp_label = torch.FloatTensor(temp_label)
                unsupervise_img.append(temp_img)
                unsupervise_text.append(temp_text)
                unsupervise_label.append(temp_label)
            return supervise_img,supervise_text, supervise_label,unsupervise_img,unsupervise_text
        elif opt.train == False:
            img = Image.open(self.testimgrecordlist[index]).convert('RGB').resize((256, 256))
            text = self.testtextlist[index]
            label = self.testlabellist[index]
            img = transforms.ToTensor()(img).float()
            text = torch.FloatTensor(text)
            label = torch.FloatTensor(label)
            return img,text,label

    def __len__(self):
        if opt.train == True and opt.supervise == True:
            return len(self.superviselabellist)
        elif opt.train == True and opt.supervise == False:
            return int(len(self.unsuperviselabellist) / self.pro2[1])
        elif opt.train == False:
            return len(self.testlabellist)
