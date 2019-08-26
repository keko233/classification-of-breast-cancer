# -*- coding: utf-8 -*-
from imutils import paths
from keras.models import load_model
from skimage import io
import cv2
import numpy as np
import os
#file_dir = '/cptjack/totem/yatong/all_data/mil_data_512/train/1/b005'
#file_paths = list(paths.list_images(file_dir))
#print(file_paths)
#
#model_path = '/cptjack/totem/yatong/4_classes/resnet50_0725/resnet50(224).h5'
#model = load_model(model_path)
#for p in file_paths:
#    n = p.split('/')[-1]
#    print(n)
#    img = io.imread(p)
#    img1 = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
#    img1 = img1.reshape(1, 224, 224, 3)
#    output = model.predict(img1/255.0)
#    pro = output[0][1]
#    print(n, pro)
#    
#    
#file = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/Photos/Benign/b003.tif'
def get_patch(file, result_dir):
    f = file.split('/')[-1]
    f_name = f.split('.')[-2]
    img = io.imread(file)
    img = np.asarray(img)
    step = 512
    step2 = 128
#    step2 = 1
#    print(img.shape)
#    h_count = img.shape[0] // step
#    w_count = img.shape[1] // step
    h_count = ((img.shape[0] - step) // step2 ) + 1
    w_count = ((img.shape[1] - step) // step2 ) + 1
    i = 0 
    print(f_name,h_count, w_count)
    
    for y in range(h_count):
        for x in range(0,w_count):
            x0 =  x * step2
            x1 = x0 + step
            y0 = y * step2
            y1 = y0 + step
#            print(x0,x1,y0,y1)
            patch = img[y0:y1, x0:x1]
            rgb_s = (abs(patch[:,:,0] -107) >= 93) & (abs(patch[:,:,1] -107) >= 93) & (abs(patch[:,:,2] -107) >= 93)
            if np.sum(rgb_s)<=(step * step ) * 0.4:
                i = i + 1
                io.imsave(result_dir+ '/'+f_name + '_'+str(i) +'.png', patch)
    return
#result_dir = '/cptjack/totem/yatong/all_data/a_test/'
#get_patch(file, result_dir)
data_dir = '/cptjack/totem/yatong/all_data/bach_test_augment'
result_dir = '/cptjack/totem/yatong/all_data/bach_test_augment_512'
for i in range(18,100):
    data_dir2 = os.path.sep.join([data_dir , str(i)])
    data_paths = list(paths.list_images(data_dir2))
    result_dir2 = os.path.sep.join([result_dir, str(i)])
    if not os.path.exists(result_dir2):os.makedirs(result_dir2)
    for file in data_paths:
        get_patch(file,result_dir2)

















