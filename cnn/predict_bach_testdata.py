# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.models import Model
import numpy as np
import os
import tensorflow as tf
from imutils import paths
from skimage import io
import cv2
#import csv

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options = tf.GPUOptions(allow_growth = True)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

cnn_model_path = '/cptjack/totem/yatong/4_classes/resnet50_0725/resnet50(224).h5'
#cnn_model_path = '/cptjack/totem/yatong/new_data/inceptionv3/inceptionv3.h5'

#lstm_model_path = '/cptjack/totem/yatong/4_classes/resnet50(0725)_lstm(0730)/lstm.h5'
cnn_model = load_model(cnn_model_path)
#layer_model = Model(inputs = cnn_model.input, outputs = cnn_model.layers[-3].output)
#lstm_model = load_model(lstm_model_path)

test_data_dir = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Test/Photos'
#test_data_dir = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/Photos/Invasive'
test_data_paths = list(paths.list_images(test_data_dir))


            
def get_predict(file, pred):
    n = file.split('/')[-1]
    img = io.imread(file)
#    img = np.array(img)
    step = 512
    h_count = img.shape[0] // step
    w_count = img.shape[1] // step
    temp = []
    data = []
#    print(h_count, w_count)
    for y in range(h_count):
        for x in range(w_count):
            x0 = x * step
            x1 = x0 + step
            y0 = y * step
            y1 = y0 + step
#            print(x0, x1, y0, y1)
            patch = img[y0:y1, x0:x1]
#            patch = np.asarray(patch)
            img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
            img1 = img1.reshape(1, 224, 224, 3)
            output = cnn_model.predict(img1/255.0)
            print(output)
            p = np.argmax(output, axis = 1)
            temp.append(p[0])
            count_0 = temp.count(0.)
            count_1 = temp.count(1.)
            count_2 = temp.count(2.)
            count_3 = temp.count(3.)
    count = [count_0, count_1, count_2, count_3]
    print(count)
    preIndex = np.argmax(count)
#            temp.append(output[0])
#    data.append(temp)
#    data = np.array(data)
#    a = []
#    a.append(data)
#    pro = lstm_model.predict(a)
#    preIndex = np.argmax(pro, axis = 1)
    pred[n] = preIndex
    return pred

pred = {}
for file in test_data_paths:
    pred = get_predict(file, pred)
print(pred)

#file = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Test/Photos/test1.tif'
#p = get_predict(file, pred)
#print(p)

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
#import keras_efficientnets
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#model_path = '/cptjack/totem/yatong/new_data/resnet50_1/resnet50_2(224).h5'
#model_path = '/cptjack/totem/yatong/new_data/inceptionv3_3/inceptionv3.h5'
#model_path = '/cptjack/totem/yatong/new_data/EfficientNet_1/EfficientNet.h5'
#model_path = '/cptjack/totem/yatong/4_classes/resnet50_0725/resnet50(224).h5'
model_path = '/cptjack/totem/yatong/4_classes/inceResV2_0806_3/InceptionResnetV2(224).h5'
model = load_model(model_path)
#model.summary()
#print(model.layers[-3])
layer_model = Model(inputs = model.input, outputs = model.layers[-2].output)
#model_path2 = '/cptjack/totem/yatong/new_data/lstm_3(0705)/lstm.h5'
#model_path2 = '/cptjack/totem/yatong/4_classes/resnet50(0725)_lstm(0730)/lstm.h5'
model_path2 = '/cptjack/totem/yatong/4_classes/inceptionResnetV2_0806_3_lstm(0807)/lstm_2.h5'
lstm_model = load_model(model_path2)
print('1')     
def get_one_img(file): 
    
    l = file.split('/')[-1]
    f = l.split('.')[-2]
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
            x0 =  x * step
            x1 = x0 + step
            y0 = y * step
            y1 = y0 + step
            patch = img[y0:y1, x0:x1]
            i = i + 1
#                io.imsave(result_dir + '/' + l +'_'+ f_name + '_'+str(i) +'.png', patch)
            img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
            img1 = img1.reshape(1,224,224,3)
            output = layer_model.predict(img1/255.0)
            temp.append(output[0])
#    print(len(temp))
    data.append(temp)
        
#        data = np.asarray(data)
#        p = lstm_model.predict(data)
#        proIndex = np.argmax(p, axis = 1)
#        print(f,'class1:', proIndex)
    return data,f


base_dir = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Test/Photos'
base_paths = list(paths.list_images(base_dir))
#benign_path = os.path.sep.join([base_path, 'Benign'])
#insitu_path = os.path.sep.join([base_path, 'InSitu'])
#normal_path = os.path.sep.join([base_path, 'Normal'])
#invasive_path = os.path.sep.join([base_path,'Invasive'])

#
#class0_path1 = list(paths.list_images(benign_path))
#class0_path2 = list(paths.list_images(insitu_path))
#class0_path3 = list(paths.list_images(normal_path))

#class0_paths = class0_path1 + class0_path2 + class0_path3
#class1_paths = list(paths.list_images(invasive_path))

#class0_data = get_all_imgdata(class0_paths)
#class1_data = get_all_imgdata(layer_model, lstm_model, class1_paths)
a = []

for file in base_paths:
#    name = file.split('/')[-1]
#    print(file)
#    if file.split('.')[-1] == 'tif':continue
    
    data, f = get_one_img(file)
    data = np.asarray(data)
    p = lstm_model.predict(data)
    proIndex = np.argmax(p, axis = 1)
#    if proIndex[0] == 0:
#    print('class3:', f, proIndex)
    a.append((f,proIndex[0]))
a.sort()
print(len(a),':', a)
#file = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Test/Photos/test1.tif'
#p = get_predict(file, pred)
#print(p)