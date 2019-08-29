# -*- coding: utf-8 -*-
'''
将2048x1536图片切割为9张512x512的patch
按顺序提取patch的cnn瓶颈特征，将提取得到的9维cnn特征作为对应2048x1536图片的特征
'''
from utils import get_img_data as g
import tensorflow as tf
from imutils import paths
import numpy as np
from keras.models import Model
from keras.models import load_model
import os
import random



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model_path = '/cptjack/totem/yatong/4_classes/inceResV2_0806_3/InceptionResnetV2(224).h5'

model = load_model(model_path)
#得到cnn的layer model，用于后续的特征提取
layer_model = Model(inputs = model.input, outputs = model.layers[-2].output)

#需要提取cnn特征用于后续lstm训练的数据集路径
train_class0 = '/cptjack/totem/yatong/all_data/bach_augment_data/0'
train_class1 = '/cptjack/totem/yatong/all_data/bach_augment_data/1'
train_class2 = '/cptjack/totem/yatong/all_data/bach_augment_data/2'
train_class3 = '/cptjack/totem/yatong/all_data/bach_augment_data/3'

#得到数据集中每张图片的路径列表
train_class0_paths = list(paths.list_images(train_class0))
train_class1_paths = list(paths.list_images(train_class1))
train_class2_paths = list(paths.list_images(train_class2))
train_class3_paths = list(paths.list_images(train_class3))

#调用get_img_data脚本的中的get_all_imgdata函数，得到最后的lstm训练数据。
#数据以列表的形式保存
class0_traindata = g.get_all_imgdata(train_class0_paths, layer_model)
class1_traindata = g.get_all_imgdata(train_class1_paths, layer_model)
class2_traindata = g.get_all_imgdata(train_class2_paths, layer_model)
class3_traindata = g.get_all_imgdata(train_class3_paths, layer_model)

#将四个列表合并为一个列表
train_data = class0_traindata + class1_traindata + class2_traindata + class3_traindata

#得到对应的标签列表
train_labels = [0 for i in class1_traindata]  + [1 for i in class0_traindata] + [2 for i in class1_traindata]  + [3 for i in class0_traindata]

#将训练集打乱
train = list(zip(train_data, train_labels))
print(len(train))
random.seed(62)
random.shuffle(train)
train_data, train_labels = zip(*train)

#保存训练集的路径
save_dir ='/cptjack/totem/yatong/4_classes/lstm_data'

#分别保存训练集数据以及标签
np.save(save_dir + '/train_data(inceptionResnetV2_0806_3).npy', train_data)
np.save(save_dir + '/train_labels(inceptionResnetV2_0806_3).npy', train_labels)

