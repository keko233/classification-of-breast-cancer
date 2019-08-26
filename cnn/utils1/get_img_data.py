# -*- coding: utf-8 -*-
"""
split 2048x1536 image into 9 small 512x512 patch
using cnn'dense layer model predict 16dim feature 
temp = 9 small patch's 16dim feature
data = all image's temp
"""

from imutils import paths
import numpy as np
import cv2
from PIL import Image 
from keras.models import Model
from keras.models import load_model
import os
from PIL import Image
from skimage import io

#def get_all_imgdata(Paths, layer_model):
#    data = []
#    for file in Paths:
#        if file.split('.') == 'tif':continue
#        l = file.split('/')[-2]
#        print(l)
##        f = file.split('/')[-1]
##        f_name = f.split('.')[-2]
#        img = Image.open(file)
#        img = np.asarray(img)
#        step = 512
##        shape = img.shape
##        if img.shape[0] < img.shape[1]:
#        h_count = img.shape[0] // step
#        w_count = img.shape[1] // step
#        i = 0 
#        temp = []
#        for y in range(h_count):
#            for x in range(0,w_count-1):
#                x0 = 256 + x * step
#                x1 = x0 + step
#                y0 = y * step
#                y1 = y0 + step
#    #            print(x0,x1,y0,y1)
#                patch = img[y0:y1, x0:x1]
#                i = i + 1
##                io.imsave(result_dir + '/' + l +'_'+ f_name + '_'+str(i) +'.png', patch)
#                img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
#                img1 = img1.reshape(1,224,224,3)
#                output = layer_model.predict(img1/255.0)
#                temp.append(output[0])
#        print(len(temp))
#        data.append(temp)
#    return data

def get_all_imgdata(Paths, layer_model):
    data = []
    for file in Paths:
        if file.split('.') == 'tif':continue
        l = file.split('/')[-2]
        f = file.split('/')[-1]
        print(f)
        img = Image.open(file)
        img = np.asarray(img)
        step = 512
#        shape = img.shape
        if img.shape[0] < img.shape[1]:
            print('1:',img.shape, f)
            h_count = img.shape[0] // step
            w_count = img.shape[1] // step
            i = 0 
            temp = []
            for y in range(h_count):
                for x in range(0,w_count-1):
                    x0 = 256 + x * step
                    x1 = x0 + step
                    y0 = y * step
                    y1 = y0 + step
#                    print(x0,x1,y0,y1)
                    patch = img[y0:y1, x0:x1]
                    i = i + 1
#                    io.imsave(result_dir + '/' + l +'_'+ f_name + '_'+str(i) +'.png', patch)
                    img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
                    img1 = img1.reshape(1,224,224,3)
                    output = layer_model.predict(img1/255.0)
                    temp.append(output[0])
            print(len(temp))
            data.append(temp)
        else:
            print('2:',img.shape, f)
            h_count = img.shape[0] // step
            w_count = img.shape[1] // step
            i = 0 
            temp = []
            for y in range(0,h_count - 1):
                for x in range(w_count):
                    x0 = x * step
                    x1 = x0 + step
                    y0 = 256 + y * step
                    y1 = y0 + step
#                    print(x0,x1,y0,y1)
                    patch = img[y0:y1, x0:x1]
                    i = i + 1
#                    io.imsave(result_dir + '/' + l +'_'+ f_name + '_'+str(i) +'.png', patch)
                    img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
                    img1 = img1.reshape(1,224,224,3)
                    output = layer_model.predict(img1/255.0)
                    temp.append(output[0])
            print(len(temp))
            data.append(temp)
          
         
    return data