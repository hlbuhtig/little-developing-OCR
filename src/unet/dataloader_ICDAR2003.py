from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import scipy.io as io
from PIL import Image
import numpy as np
from list_tool import *

dirr='../../data/'


import torchvision.transforms.functional as F
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h]) # randomly choose the location when padding
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
preprocess = transforms.Compose([
    SquarePad(),#pad to square without change shapes
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    #normalize
])


def default_loader(path): 
    img_pil =  Image.open(path)
    
    img_tensor = preprocess(img_pil)
    
    return img_tensor

def setf(wh):#in,out; return list
    di=getlist(wh)
    fs=di['d']
    tg=di['m']
    return fs,tg



class trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.images ,self.target=setf(dirr+'/SceneTrialTrain/')
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        tn= self.target[index]
        img2= self.loader(tn)
        return img,img2

    def __len__(self):
        return len(self.images)
        
class testset(Dataset):
    def __init__(self, loader=default_loader):
        self.images ,self.target=setf(dirr+'/SceneTrialTest/')
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        tn= self.target[index]
        img2= self.loader(tn)

        return img,img2

    def __len__(self):
        return len(self.images)
