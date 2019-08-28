# -*- coding: utf-8 -*-
"""
2019-08-14
预测2048x1536图片（bach challenge 的测试集）对应的label
1、将2048x1536图片重叠为256的切割为512x512的patch
2、将patch输入模型中进行预测，记录预测结果
3、投票计算一张2048x1536图片中的patch被预测为最多的label是哪个，该label为这张大图的label
"""

from imutils import paths
import numpy as np
import cv2
from PIL import Image 
from keras.models import Model
from keras.models import load_model
import os
from skimage import io
import tensorflow as tf
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model_path = '/cptjack/totem/yatong/4_classes/resnet50_newhsv_0813/resnet50.h5'
model = load_model(model_path)

data_dir = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Test/Photos'
data_files = os.listdir(data_dir)
print(data_files)
'''
model:预测模型
data_dir:保存需要预测的图片的路径
输出：打印预测的图片名，以及该图片对应的预测结果
'''
def get_top(model, data_dir): 
    #本项目中需要预测的test图片共100张，且命名有规律，故循环100次得到文件路径名。可自行更改。
    for i in range(100):
        n = 'test'+ str(i) + '.tif'
        data_path = os.path.sep.join([data_dir, n])
        print(data_path)
        img = io.imread(data_path)
        img = np.uint8(img)
        step = 512
        step2 = 256
        h_count = ((img.shape[0] - step) // step2) + 1
        w_count = ((img.shape[1] - step) // step2) + 1
        pre_list = []
        i = 0
        for y in range(h_count):
            for x in range(w_count):
                i = i+1
                patch = img[y * step2:(y * step2)+ step, x * step2:(x * step2) + step]
                img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
                img2 = img1.reshape(1,224,224,3)
                output = model.predict(img2/255.0)
                preIndex = np.argmax(output)
                #保存每张patch的预测结果label
                pre_list.append(preIndex)
        #分别计算保存的结果label中出现0、1、2、3的次数
        count_0 = pre_list.count(0.)
        count_1 = pre_list.count(1.)
        count_2 = pre_list.count(2.)
        count_3 = pre_list.count(3.)
        count = [count_0, count_1, count_2, count_3]
        #取出现次数最多的label作为该大图的预测结果
        preIndex = np.argmax(count)
        #输出图片名，以及对应的预测结果
        print(n, preIndex)
        gc.collect()
    return

get_top(model,data_dir)