import os
import glob
import cv2
import PIL
from PIL import Image
import numpy as np
import pandas as pd

f1 = sorted(os.listdir("new"))

train = []
test = []
val = []

os.makedirs('train')
os.makedirs('test')
os.makedirs('val')

for f in f1:
        f_2 = sorted(os.listdir("new"+"/"+f))
	os.makedirs(os.path.join('train',f))
	os.makedirs(os.path.join('test',f))
	os.makedirs(os.path.join('val',f))
        for x in f_2:
                path = "new"+"/"+f+"/"+x
                train_path = "train"+"/"+f
                test_path = "test"+"/"+f
                val_path = "val"+"/"+f
                
                files = sorted(os.listdir(path))
		if files > 1000:
			os.makedirs(os.path.join(train_path,x)
			os.makedirs(os.path.join(test_path,x)
			os.makedirs(os.path.join(val_path,x)
	                print(f+":",x+":",len(files))
