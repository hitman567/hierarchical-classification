import numpy as np
import pandas as pd
import os

text_file = open("list_category_cloth.txt", "r")
lines = text_file.readlines()
text_file.close()

del(lines[0:2])

x=[]
y=[]
for i in lines:
  # print(i)
  x.append(i.split()[0])
  y.append(i.split()[1])

os.makedirs('new')

i=0;
for subfolder_name in x:
  # print(subfolder_name)
  if y[i] == '1':
    os.makedirs(os.path.join('new'+'/'+'1', subfolder_name))
  elif y[i] == '2':
    os.makedirs(os.path.join('new'+'/'+'2', subfolder_name))
  else:
    os.makedirs(os.path.join('new'+'/'+'3', subfolder_name))
  i=i+1
