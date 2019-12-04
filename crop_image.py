import os
import PIL
from PIL import Image 
import cv2
PIL.Image.MAX_IMAGE_PIXELS = 933120000
f1=open("list_category_cloth.txt","r")
xx=f1.readlines()
del(xx[0:2])
dic={}
for a in xx:
    a=a.split()
    dic.update({a[0]:a[1]})

f=open("list_bbox.txt","r")
for x in f:
    y=x.split("/")
    z=y[2].split()
    print(y[2])
    print(y[1])
    img=Image.open(y[0]+"/"+y[1]+"/"+z[0])
    img=img.convert('RGB')
    img = img.crop((int(z[1]), int(z[2]),int( z[3]), int(z[4])))
    f_n=y[1].split("_")
    f_name=f_n[-1]
    path="new/"+dic[f_name]+"/"+f_name
    index=len( os.listdir(path))+1
    print(path+"/"+dic[f_name]+"_"+y[1]+"_"+z[0])
    print(img)
    img.save(path+"/"+dic[f_name]+"_"+y[1]+"_"+z[0])
