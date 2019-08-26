#-*- coding: utf-8 -*-

import numpy as np
from skimage import io
import cv2
import os
#file = '/cptjack/totem/yatong/bach_svs_result/cnn_result/result_map/test10resnet50_0725_(0802)_map.png'
#file = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/WSI/gt_thumbnails/A07.png'



#
#file = '/cptjack/totem/yatong/all_data/mil_data_512/train/3'
##os.rmdir(file)
##from imutils import paths
#import shutil
#shutil.rmtree(file)

#im1=np.ones([2,2])
#im1[0][1]=-1
#im1[1][0]=1
#print(im1)
#/cptjack/totem/yatong/bach_svs_result/InceptionResnetV2_0806_3/result_map/4_classes_map/test1_InceptionResnetV2_0806_3_map.png
pixels_list = [(10663, 9398), (13782, 11223), (8798, 8344), (15738, 8531), (14262, 9062), (12711, 6993), (14096, 10396), (9995, 7497), (12906, 10935), (14483, 10928)]
#file_dir = '/cptjack/totem/yatong/bach_svs_result/InceptionResnetV2_0806_3/result_map/4_classes_map'
#file_dir = '/cptjack/totem/yatong/bach_svs_result/lstm_result/InceptionResnetV2_0806_3_lstm/result_map/map'
#file_dir = '/cptjack/totem/yatong/bach_svs_result/cnn_result/mil_res50(0807)_epoch3/result_map/map'
file_dir = '/cptjack/totem/yatong/bach_svs_result/resnet50_newhsv_0813/result_map/map'
files = os.listdir(file_dir)
#files = [int(i.split('_')[0][4:6]) for i in files]
#print(files)
files.sort(key = lambda a:int(a.split('_')[0][4:6]))
print(files)
for index, file in enumerate(files):
    index2 = index + 1
    file_paths = os.path.sep.join([file_dir, file])
    print(index, file_paths)
    print(pixels_list[index])
    img = io.imread(file_paths)
    out_img = cv2.resize(img, (pixels_list[index]), interpolation=cv2.INTER_AREA)
    io.imsave('/cptjack/totem/yatong/bach_svs_result/temp/' + str(index2) + '.png' ,out_img)

#img = io.imread(file)
#out_img = cv2.resize(img, (int(10663), int(9398)), interpolation=cv2.INTER_AREA)
#io.imsave('/cptjack/totem/yatong/bach_svs_result/temp/1.png',out_img)
#img = np.int8(img)


#    
#    np.save(, dir0_top)
#    np.save(result_dir+'/dir1_top.npy', dir1_top)
#    np.save(result_dir+'/dir2_top.npy', dir2_top)
#    np.save(result_dir+'/dir3_top.npy', dir3_top)

#result_dir = '/cptjack/totem/yatong/all_data/mil_data'
#dir0 = np.load(result_dir+'/dir0_top.npy')
#dir1 = np.load(result_dir+'/dir1_top.npy')
#dir2 = np.load(result_dir+'/dir2_top.npy')
#dir3 = np.load(result_dir+'/dir3_top.npy')
#print(dir3[-5])
#
##for i in range(100):
###    print(i)
##    print('1')
#from skimage import io
#import cv2
#file = '/cptjack/totem/yatong/all_data/bach_augment_data_512/train/0/n001_2_lr_24.png'
#img = io.imread(file)
#io.imshow(img)
#img1 = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
#cv2.imwrite(result_dir + '/1.png', img)
#io.imshow(img1)
#img2 = cv2.imread(file)
#io.imshow(img2)

#name0_1 = np.load('/cptjack/totem/yatong/all_data/mil_data/07298_name1.npy')
#name0_0 = np.load('/cptjack/totem/yatong/all_data/mil_data/07299_name1.npy')
#print(name0_1[0:5])
#print(name0_0[0:5])
#print((name0_1[6]))
#print((name0_0[6]))

#data = np.load('/cptjack/totem/yatong/all_data/mil_data/0731/3/0_data.npy')
#print(data[-1][0:16])
##print(data[0:16])