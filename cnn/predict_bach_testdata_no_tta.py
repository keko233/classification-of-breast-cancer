# -*- coding: utf-8 -*-
"""
2019-08-14
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

def get_top(model, data_dir): 
    for i in range(100):
        n = 'test'+ str(i) + '.tif'
        data_path = os.path.sep.join([data_dir, n])
#    for p in data_dir_list:
        print(data_path)
#        f = p.split('/')[-1]
#        n = f.split('.')[-2]
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
#                n1 = n + '_' + str(i)
#                x0 = x * step2
#                x1 = x0 + step
#                y0 = y * step2
#                y1 = y0 + step
#                patch = img[y0:y1, x0:x1]
                patch = img[y * step2:(y * step2)+ step, x * step2:(x * step2) + step]
                img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
                img2 = img1.reshape(1,224,224,3)
                output = model.predict(img2/255.0)
                preIndex = np.argmax(output)
#        print(preIndex)
                pre_list.append(preIndex)
#        del img, img1
        count_0 = pre_list.count(0.)
        count_1 = pre_list.count(1.)
        count_2 = pre_list.count(2.)
        count_3 = pre_list.count(3.)
        count = [count_0, count_1, count_2, count_3]
#    print(count)
        preIndex = np.argmax(count)
        print(n, preIndex)
        gc.collect()
    return

get_top(model,data_dir)