# -*- coding: utf-8 -*-
"""
predict bach_data 2048x1536 images 
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

#model_path = '/cptjack/totem/yatong/4_classes/resnet50_0725/resnet50(224).h5'
#model_path = '/cptjack/totem/yatong/4_classes/inceResV2_0806_3/InceptionResnetV2(224).h5'
model_path = '/cptjack/totem/yatong/4_classes/mil_resnet50_0807/3_resnet50(224).h5'
model = load_model(model_path)

data_dir = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Test/Photos'
for i in range(57,100):
    data_dir2 = os.path.sep.join([data_dir, str(i)])
    data_paths = list(paths.list_images(data_dir2))
    pre_list = []
    n1 = data_paths[0].split('/')[-1]
    n2 = n1.split('_')[0]
#    print(n2)
    for p in data_paths:
        img = io.imread(p)
        img1 = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        img1 = img1.reshape(1,224,224,3)
        prob = model.predict(img1/255.0)
#        print(prob)
        preIndex = np.argmax(prob)
#        print(preIndex)
        pre_list.append(preIndex)
        del img, img1
    count_0 = pre_list.count(0.)
    count_1 = pre_list.count(1.)
    count_2 = pre_list.count(2.)
    count_3 = pre_list.count(3.)
    count = [count_0, count_1, count_2, count_3]
#    print(count)
    preIndex = np.argmax(count)
    print(n2, preIndex)
    gc.collect()

