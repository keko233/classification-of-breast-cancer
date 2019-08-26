# -*- coding: utf-8 -*-
from utils1 import get_img_data as g
import tensorflow as tf
from imutils import paths
import numpy as np
from keras.models import Model
from keras.models import load_model
import os
from skimage import io
import random
#import keras_efficientnets


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#model_path = '/cptjack/totem/yatong/new_data/inceptionResnetV2(0716)/inceptionResnetV2.h5'
#model_path = '/cptjack/totem/yatong/4_classes/resnet50_0725/resnet50(224).h5'
model_path = '/cptjack/totem/yatong/4_classes/inceResV2_0806_3/InceptionResnetV2(224).h5'

model = load_model(model_path)
model.summary()
print(model.layers[-2])
layer_model = Model(inputs = model.input, outputs = model.layers[-2].output)

#train_class0 = '/cptjack/totem/yatong/all_data/bach_augment_data/0'
#train_class1 = '/cptjack/totem/yatong/all_data/bach_augment_data/1'
#train_class2 = '/cptjack/totem/yatong/all_data/bach_augment_data/2'
#train_class3 = '/cptjack/totem/yatong/all_data/bach_augment_data/3'

train_class0 = '/cptjack/totem/yatong/all_data/balance_dataset/train/0'
train_class1 = '/cptjack/totem/yatong/all_data/balance_dataset/train/1'
train_class2 = '/cptjack/totem/yatong/all_data/balance_dataset/train/2'
train_class3 = '/cptjack/totem/yatong/all_data/balance_dataset/train/3'

train_class0_paths = list(paths.list_images(train_class0))
train_class1_paths = list(paths.list_images(train_class1))
train_class2_paths = list(paths.list_images(train_class2))
train_class3_paths = list(paths.list_images(train_class3))

print(len(train_class1_paths), len(train_class0_paths))

class0_traindata = g.get_all_imgdata(train_class0_paths, layer_model)
class1_traindata = g.get_all_imgdata(train_class1_paths, layer_model)
class2_traindata = g.get_all_imgdata(train_class2_paths, layer_model)
class3_traindata = g.get_all_imgdata(train_class3_paths, layer_model)

train_data = class0_traindata + class1_traindata + class2_traindata + class3_traindata
train_labels = [0 for i in class1_traindata]  + [1 for i in class0_traindata] + [2 for i in class1_traindata]  + [3 for i in class0_traindata]

train = list(zip(train_data, train_labels))
print(len(train))
random.seed(62)
random.shuffle(train)
train_data, train_labels = zip(*train)

save_dir ='/cptjack/totem/yatong/4_classes/lstm_data'
#np.save(save_dir + '/train_data(resnet50_0725).npy', train_data)
#np.save(save_dir + '/train_labels(resnet50_0725).npy', train_labels)
#np.save(save_dir + '/val_data(resnet50_0725).npy', train_data)
#np.save(save_dir + '/val_labels(resnet50_0725).npy', train_labels)
#np.save(save_dir + '/train_data(inceptionResnetV2_0806_3).npy', train_data)
#np.save(save_dir + '/train_labels(inceptionResnetV2_0806_3).npy', train_labels)
np.save(save_dir + '/val_data(inceptionResnetV2_0806_3).npy', train_data)
np.save(save_dir + '/val_labels(inceptionResnetV2_0806_3).npy', train_labels)
#print(len(val_class1_paths), len(val_class0_paths))
#class1_valdata = g.get_all_imgdata(val_class1_paths, layer_model)
#class0_valdata = g.get_all_imgdata(val_class0_paths, layer_model)
#val_data = class1_valdata + class0_valdata
#val_labels = [1 for i in class1_valdata] + [0 for i in class0_valdata]

#np.save(save_dir + '/val_data(1536 efficient_2).npy', val_data)
#np.save(save_dir + '/val_labels(1536 efficient_2).npy', val_labels)


#class1_traindata = g.get_all_imgdata(b.train_class1_paths, layer_model)
#class0_traindata = g.get_all_imgdata(b.train_class0_paths, layer_model)
#class1_valdata = g.get_all_imgdata(b.val_class1_paths, layer_model)
#class0_valdata = g.get_all_imgdata(b.val_class0_paths, layer_model)
#
#train_data = class1_traindata + class0_traindata
#train_labels = [1 for i in class1_traindata] + [0 for i in class0_traindata]
#train = list(zip(train_data, train_labels))
#random.seed(43)
#random.shuffle(train)
#train_data, train_labels = zip(*train)
#print(len(train_data))
#
#val_data = class1_valdata + class0_valdata
#val_labels = [1 for i in class1_valdata] + [0 for i in class0_valdata]
#val = list(zip(val_data, val_labels))
#random.seed(23)
#random.shuffle(val)
#val_data, val_labels = zip(*val)
#
#save_dir = '/cptjack/totem/yatong/new_data/data/train'
#save_dir2 = '/cptjack/totem/yatong/new_data/data/validation'
#np.save(save_dir + '/train_data.npy', train_data)
#np.save(save_dir + '/train_label.npy', train_labels)
#np.save(save_dir2 + '/val_data.npy', val_data)
#np.save(save_dir2 + '/val_labels.npy', val_labels)
#
#
#print(len(class0_traindata))
#print(len(class0_traindata[0]))
#print(len(class0_traindata[0][0]))
#print()
#
#base_path = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/Photos'
#benign_path = os.path.sep.join([base_path, 'Benign'])
#insitu_path = os.path.sep.join([base_path, 'InSitu'])
#normal_path = os.path.sep.join([base_path, 'Normal'])
#invasive_path = os.path.sep.join([base_path,'Invasive'])
#
#class0_path1 = list(paths.list_images(benign_path))
#class0_path2 = list(paths.list_images(insitu_path))
#class0_path3 = list(paths.list_images(normal_path))
#
#class0_paths = class0_path1 + class0_path2 + class0_path3
#class1_paths = list(paths.list_images(invasive_path))
#
#
#class1_data = g.get_all_imgdata(class1_paths, layer_model)
#class0_data = g.get_all_imgdata(class0_paths, layer_model)
#
#
#print(len(class0_data))
#save_dir = '/cptjack/totem/yatong/new_data/data'
#np.save(save_dir + '/testdata_0.npy', class0_data)
#np.save(save_dir + '/testdata_1.npy', class1_data)

#save_dir = '/cptjack/totem/yatong/new_data/data'
#np.save(save_dir + '/testdata_0.npy', train_data)
#np.save(save_dir + '/testdata_1.npy', train_labels)


