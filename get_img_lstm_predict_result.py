# -*- coding: utf-8 -*-
"""
cnn+lstm模型预测 2048x1536 图片
1、取图片中间的9张512x512的patch用cnn layer model预测并得到cnn特征
2、将得到的cnn特征输入lstm中预测得到最终的预测结果
"""
import numpy as np
import cv2
from PIL import Image 

'''
获得单张2048x1536图的cnn特征
输入:
    layer_model:用于提取cnn特征的cnn layer model
    file：需要预测的图片保存路径
输出：
    data：得到的单张图片的cnn特征
    name：输入图片名
'''
def get_one_img(layer_model, file): 
    
    l = file.split('/')[-1]
    name = l.split('.')[-2]
    img = Image.open(file)
    img = np.asarray(img)
    step = 512
    h_count = img.shape[0] // step
    w_count = img.shape[1] // step
  
    i = 0 
    temp = []
    data = []
    for y in range(h_count):
        for x in range(1,w_count):
            x0 = x * step
            x1 = x0 + step
            y0 = y * step
            y1 = y0 + step
            patch = img[y0:y1, x0:x1]
            i = i + 1
            img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
            img1 = img1.reshape(1,224,224,3)
            output = layer_model.predict(img1/255.0)
            temp.append(output[0])
    data.append(temp)
    return data,name


'''
打印出每张图片预测得到的标签结果
lstm_model：用于预测的lstm模型
data_paths:保存所有图片路径的列表
'''
def get_lstm_predict_result(lstm_model,data_paths):
    a = []
    for file in data_paths:
        data, name = get_one_img(file)
        data = np.asarray(data)
        prob = lstm_model.predict(data)
        proIndex = np.argmax(prob, axis = 1)
        a.append((name,proIndex[0]))
    a.sort(key = lambda a:int(a[0:][4:6]))
    print(len(a),':', a)
    return
