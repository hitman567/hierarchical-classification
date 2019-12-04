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

for f in f1:
	f_2 = sorted(os.listdir("new"+"/"+f))
	for x in f_2:
		path = "new"+"/"+f+"/"+x
		train_path = "train"+"/"+f+"/"+x
		test_path = "test"+"/"+f+"/"+x
		val_path = "val"+"/"+f+"/"+x
		
		files = sorted(os.listdir(path))
		print(len(files))
		for num in range(0,len(files)):
			if num<100:
				os.system("cp "+path+"/"+files[num] +" "+test_path)
			elif num>=100 and num<200:
				os.system("cp "+path+"/"+files[num] +" "+val_path)
			if num>=200 and num<2000:
				os.system("cp "+path+"/"+files[num] +" "+train_path)



