2020/11/24
This program uses pytorch and general packages.

I finished the U-net part, but the other part that changes segmentations into words doesn't work. I'm stilling working on it.

So I just submit what works on this project. I believe that the problem on the other part would be finished in a few days. 

More details and comments will be updated when I submited final code.

Here is the github of this project:
https://github.com/hlbuhtig/little-developing-OCR



In dataset part:

I found there're French labels in .xml files which may cause some error when running my program, so I did a little changes on those French. All changes are in 2 segmentation.xml files.

Changes were recorded in data/info.txt







In U-net part:

I got 0.66 acc at epoch 120. (limited on gpu and not enough knowledge about NN)

My code is based on this: milesial/Pytorch-UNet.(He uses U-net on other dataset)
https://github.com/milesial/Pytorch-UNet

Run src/unet/unet_main.py to start this part

My work:
unet_main : main program of this part.
dataloader_ICDAR2003 : rewrite dataloader
list_tool : read .xml files of the dataset and process them




update 2020/11/30

Fix the 'dirr' in src/unet/dataloader_ICDAR2003.py. Now unet_main.py is able to read data in correct path.

Add SquarePad() in src/unet/dataloader_ICDAR2003.py. Now the shapes of pictures will not be changed when resizeing.

Add some comments and delete uesless codes for testing in src/unet.

