import torchvision,torch
from torchvision import datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
##############################
batch_size = 8
learningRate = 0.03
epochs=100

############################## load dataset
from dataloader_ICDAR2003 import *

train_data=trainset()
test_data=testset()



train_loader = DataLoader(train_data, batch_size = batch_size,
                         shuffle = True, num_workers = 0)
test_loader = DataLoader(test_data, batch_size = batch_size,
                         shuffle = False, num_workers = 0)



###################################
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


from unet import UNet
net = UNet(3, 1).cuda()

optimizer = optim.RMSprop(net.parameters(), lr=learningRate, weight_decay=1e-8, momentum=0.9)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
if net.n_classes > 1:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()


########################################

from eval import eval_net

macc=0.1
for epoch in range(epochs):
    net.train()

    epoch_loss = 0
    for imgs,true_masks in train_loader:
        
        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)
        #print(imgs.shape)

        masks_pred = net(imgs)
        loss = criterion(masks_pred, true_masks)
        epoch_loss += loss.item()
        

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()


    trainscore = eval_net(net, train_loader, device)        
    testscore = eval_net(net, test_loader, device)
    
    print('epoch',epoch,':','\ntrain score:',trainscore,'\ntest  score:',testscore)

    if(testscore>macc):
        macc=testscore
        torch.save(net, 'model.pkl')
        f=open('history.txt','w')
        s='score=\n'+str(trainscore)+'\n'+str(testscore)+'\nepoch=\n'+str(epoch)
        f.write(s)
        f.close()


