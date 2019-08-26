# -*- coding: utf-8 -*-
from skimage import io
import numpy as np
import cv2
from imgaug import augmenters as iaa
from imutils import paths
import os

#file1 = '/cptjack/totem/yatong/all_data/balance_dataset/train/1/Benign 40x_Y5_1.tif'
##file = '/cptjack/totem/yatong/all_data/balance_dataset/train/1/Benign 40x_Y248_1.tif'
#file2 = '/cptjack/totem/yatong/bach_data/0/n001_1.tif'
#file = '/cptjack/totem/yatong/bach_data/0/n001_1.tif'
##x = io.imread(file1)
#x = io.imread(file1)
##io.imshow(x)
#if np.random.binomial(1,0.8):
##                x = color.rgb2hed(x)
#    x = cv2.cvtColor(np.uint8(x),cv2.COLOR_RGB2HSV)
#    scale = np.random.uniform(low = 0.85, high = 1.20) 
#    print(scale)
#    x[:,:,0] = x[:,:,0] * scale
#    x = cv2.cvtColor(np.uint8(x),cv2.COLOR_HSV2RGB)
#    io.imshow(x)
##                x = color.hed2rgb(x)
#else:
#    pass

#seq = iaa.OneOf([iaa.GaussianBlur(sigma = (0, 3.1)),
#                      iaa.Fliplr(0.5),
#                      iaa.Flipud(0.5),
#                      ])
#imglist = []
#imglist.append(x1)
#imglist.append(x2)
#images_aug = seq.augment_images(imglist)
##io.imshow(images_aug[0])
#io.imshow(images_aug[1])
#cv2.imwrite(result_dir + '/1.png', images_aug[0])
#save_dir = '/cptjack/totem/yatong/all_data/bach_augment_data/0/'  
#save_dir_1 = '/cptjack/totem/yatong/all_data/bach_augment_data/1/' 
#save_dir_2 = '/cptjack/totem/yatong/all_data/bach_augment_data/2/' 
#save_dir_3 = '/cptjack/totem/yatong/all_data/bach_augment_data/3/' 
bach_data_0 = '/cptjack/totem/yatong/new_data/lstm_data/validation/0'
bach_data_1 = '/cptjack/totem/yatong/new_data/lstm_data/validation/1'
bach_data_2 = '/cptjack/totem/yatong/new_data/lstm_data/validation/2'
bach_data_3 = '/cptjack/totem/yatong/new_data/lstm_data/validation/3'
bach_data_paths_0 = list(paths.list_images(bach_data_0))
bach_data_paths_1 = list(paths.list_images(bach_data_1))
bach_data_paths_2 = list(paths.list_images(bach_data_2))
bach_data_paths_3 = list(paths.list_images(bach_data_3))
#print(len(bach_data_paths_0))

def data_augment(Paths,save_dir):
    Paths.sort(key = lambda a:a.split('/')[-1][1:5])
    for file in Paths:
        f = file.split('/')[-1]
        n = f.split('.')[-2]
        scale_list = []
#        index = n[4:]
#        print(index)
#        save_d = os.path.sep.join([save_dir, str(index)])
#        print( file ,save_d)
        x = io.imread(file)
        io.imsave(save_dir + '/' + n + '.tif', x)
        flag = 't'
        for i in range(1,2):
#            if np.random.binomial(1,0.8):
            x1 = cv2.cvtColor(np.uint8(x),cv2.COLOR_RGB2HSV)
            while(True):
                scale = np.random.uniform(low = 0.895, high = 1.06) 
#                print(scale)
                flag = 't'
#                print(len(scale_list))
                for s in scale_list:
#                    flag = 't'
                    if abs(scale - s) < 0.025:
#                        print(scale, s)
                        flag = 'f'
                        break
#                print(scale, flag)
                if abs(scale - 1.0) >= 0.025 and flag == 't':break
            scale_list.append(scale)
            print(n, '_', i,':',scale)
            x1[:,:,0] = x1[:,:,0] * scale
            x1 = cv2.cvtColor(np.uint8(x1),cv2.COLOR_HSV2RGB)
#            io.imsave(save_dir + '/'+ n + '_' + str(i) + '.tif', x1)
            io.imsave(save_dir + '/'+ n + '_0.tif', x1)
                
#            
#        x1 = io.imread(file)
#        if np.random.binomial(1,0.2):
#            seq = iaa.OneOf([iaa.GaussianBlur(sigma = (0.8, 2.2))])
#            imglist = []
#            imglist.append(x1)
#            images_aug = seq.augment_images(imglist)
#            #io.imshow(images_aug[0])
##            io.imshow(images_aug[1])
#            io.imsave(save_d +'/' + n + '_blur.tif', images_aug[0])
    return

#data_augment(bach_data_paths_0, save_dir)
#data_augment(bach_data_paths_3, save_dir_3)
#dir_1 = '/cptjack/totem/yatong/all_data/bach_augment_data/0/_1'
#paths_1 = list(paths.list_images(dir_1))
#import os
#save_dir = '/cptjack/totem/yatong/all_data/bach_test_augment'
#test_dir = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Test/Photos'
#test_paths = list(paths.list_images(test_dir))  
save_dir0 = '/cptjack/totem/yatong/all_data/new_hsv_augment_data/0'    
save_dir1 = '/cptjack/totem/yatong/all_data/new_hsv_augment_data/1'
save_dir2 = '/cptjack/totem/yatong/all_data/new_hsv_augment_data/2'
save_dir3 = '/cptjack/totem/yatong/all_data/new_hsv_augment_data/3'
data_augment(bach_data_paths_0, save_dir0)
data_augment(bach_data_paths_1, save_dir1)     
data_augment(bach_data_paths_2, save_dir2)       
data_augment(bach_data_paths_3, save_dir3)       

#a = 0.934
#b= 0.937
#c = round(b, 2)
#print(c)
  
