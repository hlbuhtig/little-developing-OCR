from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import scipy.io as io
from PIL import Image
import numpy as np
from list_tool import *

dirr='../../data/'

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
preprocess = transforms.Compose([
    
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    #normalize
])

def make_square(im, min_size=256, fill_color=(128, 128, 128)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def default_loader(path): 
    img_pil =  Image.open(path)
    
    #img_pil = make_square(img_pil)
    img_pil=img_pil.resize((128,128))
    #print(img_pil.size,type(img_pil))
    img_tensor = preprocess(img_pil)
    return img_tensor

def setf(wh):#in,out
    di=getlist(wh)
    fs=di['d']
    tg=di['m']
    #print(len(fs),len(tg))

    return fs,tg
    #return fs,torch.tensor(tg, dtype=torch.long) 




class trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.images ,self.target=setf(dirr+'/SceneTrialTrain/')
        self.loader = loader

    def __getitem__(self, index):
        #print(index)
        fn = self.images[index]
        img = self.loader(fn)
        tn= self.target[index]
        img2= self.loader(tn)
        #target = self.target[index]
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
