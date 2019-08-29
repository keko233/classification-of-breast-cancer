# -*- coding: utf-8 -*-
"""
将 2048x1536的图片切割为9个512x512的patch
用cnn的去掉最后一层全连接层的模型预测图片，得到patch的瓶颈特征，本项目中的瓶颈特征维数为32，可自行更改
最终得到的一张2048x1536大图的特征为：9个patch的瓶颈特征--> 9 x 16 维的特征
"""
import numpy as np
import cv2
from PIL import Image 

'''
输入：
    Path:保存所有2048x1536大图的路径列表
    layer_model：用于预测的cnn layer model
输出：
    data：保存所有大图的特征的三维列表。比如有10张大图，那么返回列表维度为 10x9x16
'''
def get_all_imgdata(Paths, layer_model):
    data = []
    for file in Paths:
        if file.split('.') == 'tif':continue
        f = file.split('/')[-1]
        img = Image.open(file)
        img = np.asarray(img)
        step = 512
        #图片有2048x1536的，也有1536x2048的。由于只取图片中间的九张patch，有部分边缘需要舍弃
        #当图片为2048x1536时，舍弃左右两边的256部分
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
                    img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
                    img1 = img1.reshape(1,224,224,3)
                    output = layer_model.predict(img1/255.0)
                    temp.append(output[0])
            print(len(temp))
            data.append(temp)
        #当图片为1536x2048时，舍弃图片上下的256部分
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
                    img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
                    img1 = img1.reshape(1,224,224,3)
                    output = layer_model.predict(img1/255.0)
                    temp.append(output[0])
            print(len(temp))
            data.append(temp)
            
    return data