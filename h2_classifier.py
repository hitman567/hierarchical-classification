import os
import pandas as pd
import numpy as np
import glob
import random
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

batch_size = 16
epochs = 60

num_h1 = len([x for x in glob.glob('train/*') if os.path.isdir(x)])
num_h2 = len([x for x in glob.glob('train/*/*') if os.path.isdir(x)])
target_size = (128,128)


#os.system("!rm -rf ./tb_log")
#os.system("!rm -rf ./weights")

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#make_dir(log_filepath)
#make_dir(weights_store_filepath)

#log_num=len(os.listdir("./Logs_h1_classifier"))
#log_filepath = "./Logs_h1_classifier/tb_log_"+str(log_num+1)+"/"
log_filepath = "./Logs_h2_classifier/tb_log/"
weights_store_filepath = './weights_h2_classifier/'
train_id = '1'
model_name = 'h2_model'+train_id+'.h5'
model_path = os.path.join(weights_store_filepath, model_name)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

make_dir(log_filepath)
make_dir(weights_store_filepath)

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

def generator(batch_size, target_size, folder):

    h1_folders = sorted([x for x in glob.glob(folder+'/*') if os.path.isdir(x)])
    h1_mapping = {int(b.split('/')[-1]):a for a,b in zip(range(len(h1_folders)), h1_folders)}
    h2_folders = glob.glob(folder+'/*/*')
    h2_mapping = {b.split('/')[-1]:a for a,b in zip(range(len(h2_folders)), h2_folders)}

    dict_mapping = {h2_mapping[x.split('/')[-1]]:h1_mapping[int(x.split('/')[-2])] for x in glob.glob(folder+'/*/*')}
    
    images = glob.glob(folder+'/*/*/*.jpg')
    random.shuffle(images)

    i = 0

    while True:
        image_list = []
        h1_list = []
        h2_list = []
        for b in range(batch_size):
            if i == len(images):
                i = 0
                random.shuffle(images)
            img = Image.open(images[i])
            img = np.array(image_resize(img))
            image_list.append(img)
            h2_class = h2_mapping[images[i].split('/')[-2]]
            i += 1
            h1_list.append(dict_mapping[h2_class])
            h2_list.append(h2_class)
        h1_groundtruths = to_categorical(h1_list, len(h1_folders))
        h2_groundtruths = to_categorical(h2_list, len(h2_folders))
        input_images = np.array(image_list)
        input_images = input_images/255
        yield input_images, h2_groundtruths

generator_train = generator(batch_size, target_size, 'train')
generator_validation = generator(batch_size, target_size, 'val')

def scheduler(epoch):
  learning_rate_init = 0.003
  if epoch > 15:
    learning_rate_init = 0.0005
  if epoch > 30:
    learning_rate_init = 0.0001
  return learning_rate_init

#class LossWeightsModifier(keras.callbacks.Callback):
#  def __init__(self, alpha, beta):
#    self.alpha = alpha
#    self.beta = beta
#  def on_epoch_end(self, epoch, logs={}):
#    if epoch == 12:
#      K.set_value(self.alpha, 0.3)
#      K.set_value(self.beta, 0.7)
#    if epoch == 18:
#      K.set_value(self.alpha, 0.2)
#      K.set_value(self.beta, 0.8)
#    if epoch == 42:
#      K.set_value(self.alpha, 0.0)
#      K.set_value(self.beta, 1.0)


def BCNN_model(input_shape, num_h2):

    img_input = Input(shape=input_shape, name='input')

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    c_2_bch = Flatten(name='c2_flatten')(x)
    c_2_bch = Dense(512, activation='relu', name='c2_fc_1')(c_2_bch)
    c_2_bch = BatchNormalization()(c_2_bch)
    c_2_bch = Dropout(0.5)(c_2_bch)
    c_2_bch = Dense(512, activation='relu', name='c2_fc2')(c_2_bch)
    c_2_bch = BatchNormalization()(c_2_bch)
    c_2_bch = Dropout(0.5)(c_2_bch)
    c_2_pred = Dense(num_h2, activation='softmax', name='h2_prediction')(c_2_bch)


    model = Model(inputs=img_input, outputs=[c_2_pred], name='hierarchical_classifier')

    return(model)

hierarchical_model = BCNN_model((target_size[0], target_size[1], 3), num_h2)

sgd = optimizers.SGD(lr=0.003, momentum=0.9, nesterov=True)

hierarchical_model.compile(loss='categorical_crossentropy', 
              optimizer=sgd,
              metrics=['accuracy'])

tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr, tb_cb]

spe = len(glob.glob('train/*/*/*.jpg')) // batch_size
vs = len(glob.glob('val/*/*/*.jpg')) // batch_size

hierarchical_model.fit_generator(generator_train, steps_per_epoch=spe, epochs=epochs, callbacks=cbks,
                                 validation_data=generator_validation,
                                 validation_steps=vs,verbose=2)

hierarchical_model.compile(loss='categorical_crossentropy',
            optimizer=sgd, 
            metrics=['accuracy'])

hierarchical_model.save(model_path)

