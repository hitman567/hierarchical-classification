import os
import pandas as pd
import numpy as np
import glob
import random
from sklearn.metrics  import confusion_matrix, classification_report
from PIL import Image, ImageOps
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
from keras.initializers import he_normal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.models import load_model

target_size = (128,128)
h1_folders = sorted([x for x in glob.glob('test/*') if os.path.isdir(x)])
h1_mapping = {int(b.split('/')[-1]):a for a,b in zip(range(len(h1_folders)), h1_folders)}
h2_folders = glob.glob('test/*/*')
h2_mapping = {b.split('/')[-1]:a for a,b in zip(range(len(h2_folders)), h2_folders)}
dict_mapping = {h2_mapping[x.split('/')[-1]]:h1_mapping[int(x.split('/')[-2])] for x in glob.glob('test/*/*')}

images = glob.glob('test/*/*/*.jpg')

image_list = []
h1_list = []
h2_list = []

def image_resize(img):
        ratio = target_size[0]/max(img.size)
        w = int(ratio*img.size[0])
        h = int(ratio*img.size[1])
        img = img.resize((w,h), Image.ANTIALIAS)
        new_img = Image.new("RGB", target_size)

        delta_w = target_size[0] - img.size[0]
        delta_h = target_size[1] - img.size[1]
        
        new_img.paste(img,(int(delta_w/2),int(delta_h/2)))
        
        padding = (int(delta_w/2),int(delta_h/2),int(delta_w-delta_w/2),int(delta_h-delta_h/2))
        new_img = ImageOps.expand(img, padding, fill='black')
        
        new_img = new_img.resize(target_size, Image.ANTIALIAS)
        return new_img

for i in range(len(images)):
  img = Image.open(images[i])
  img = np.array(image_resize(img))
  image_list.append(img)
  h2_class = h2_mapping[images[i].split('/')[-2]]
  h1_list.append(dict_mapping[h2_class])
  h2_list.append(h2_class)

h1_groundtruths = to_categorical(h1_list, len(h1_folders))
h2_groundtruths = to_categorical(h2_list, len(h2_folders))

input_images = np.array(image_list)
input_images = input_images/255

model = load_model('./weights/'+'hierarchical_model2.h5')

print(model.metrics_names)
score = model.evaluate(input_images,[h1_groundtruths, h2_groundtruths], verbose=0)
print(score)

pred = model.predict(input_images)

#model.summary()

l1 = pred[0]
l2 = pred[1]

result = []


#print(h1_mapping)
#print(h2_mapping)

for i in range(len(l1)):
	y=[]
	y.append(images[i])
	y1=[]
	y.append(images[i].split('/')[1])
	y.append(images[i].split('/')[2])
	for key,value in h1_mapping.items():
		if value == np.argmax(l1[i]):
			y.append(key)
	for key,value in h2_mapping.items():
		if value == np.argmax(l2[i]):
			y.append(key)
	result.append(y)

data = pd.DataFrame(result)
data.columns = ['image_name','h1_groundtruths','h2_groundtruths','h1_prediction','h2_prediction']

data.to_csv('results.csv',sep='\t', encoding='utf-8')

h1=[]
h2=[]
for i in range(1300):
	h1.append(h1_mapping[data['h1_prediction'][i]])
	h2.append(h2_mapping[data['h2_prediction'][i]])

h1_pred = to_categorical(h1,3)
h2_pred = to_categorical(h2,13)
#print(h2_mapping)
#print(h1_pred)
#print(data['h2_prediction'][0])
print('Confusion Matrix for h1 layer:')
print(confusion_matrix(h1_list,h1))
print('Classification Report for h1 layer:')
print(classification_report(h1_list,h1))

print('Confusion Matrix for h2 layer:')
print(confusion_matrix(h2_list,h2))
print('Classification Report for h2 layer:')
print(classification_report(h2_list,h2))

print(h2_mapping)
#print(l1)
#print(l2[0])
#print(len(pred))
#print(pred[0].shape)

#print(data)
