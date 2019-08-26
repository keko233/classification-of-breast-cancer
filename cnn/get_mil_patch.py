# -*- coding: utf-8 -*-
from keras.models import load_model
from skimage import io
import cv2
import tensorflow as tf
import mil_data as m
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#    model_path = '/cptjack/totem/yatong/4_classes/resnet50_0724/resnet50(224).h5'
#    model = load_model(model_path)

#print(tf.__version__)
def get_top_patch(model, dir_files):
    all_top_4_list = []
    for files in dir_files:
        prob_list = []
        print(prob_list)
    #    print(len(files))
        for f in files:
            img = io.imread(f)
            img1 = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
            img1 = img1.reshape(1,224,224,3)
            prob = model.predict(img1/255.0)
            
            prob_list.append(prob[0][0])
        print(prob_list)
        data = list(zip(prob_list,files ))
        data.sort(reverse = True)
        prob_list, files = zip(*data)
        print(prob_list)
        files = list(files)
        top_4_list = files[0:4]
        print(top_4_list)
        all_top_4_list.append(top_4_list)
    print(len(all_top_4_list))
    return all_top_4_list

def get_all_top_patch(model_path, dir_files):
    model = load_model(model_path)
    dir0_top = get_top_patch(model,dir_files[0])
    dir1_top = get_top_patch(model,dir_files[1])
    dir2_top = get_top_patch(model,dir_files[2])
    dir3_top = get_top_patch(model,dir_files[3])
    return dir0_top, dir1_top, dir2_top, dir3_top
