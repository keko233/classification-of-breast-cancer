# -*- coding: utf-8 -*-
"""
get the transformation image of 40x image(2048x1536):
                                        rotate it 90,180, 270 degrees.
                                        flip it up and down
                                        flip it left and right
increase one image to 6 images.
"""
#import build_dataset3 as b
from PIL import Image
import os
from imutils import paths
#from imgaug import augmenters as iaa
import numpy as np
#print(len(b.val_class0_paths))
#save_base_dir = '/cptjack/totem/yatong/new_data/lstm_data'
#train_dir = os.path.sep.join([save_base_dir, 'train'])
#val_dir = os.path.sep.join([save_base_dir, 'validation'])
#train_class1_dir = os.path.sep.join([train_dir, '1'])
#train_class0_dir = os.path.sep.join([train_dir, '0'])
#val_class1_dir = os.path.sep.join([val_dir, '1'])
#val_class0_dir = os.path.sep.join([val_dir, '0'])
#if not os.path.exists(train_class1_dir):os.makedirs(train_class1_dir)
#if not os.path.exists(train_class0_dir):os.makedirs(train_class0_dir)
#if not os.path.exists(val_class1_dir):os.makedirs(val_class1_dir)
#if not os.path.exists(val_class0_dir):os.makedirs(val_class0_dir)
#path_0='/cptjack/totem/yatong/bach_data_color_norm/train/new0'
#path_1='/cptjack/totem/yatong/bach_data_color_norm/train/new1'
#save_dir_1 = '/cptjack/totem/yatong/bach_data_color_norm/train/g1/'
#save_dir_0 = '/cptjack/totem/yatong/bach_data_color_norm/train/g0/'
def get_transform(Paths, save_dir):
    for file in Paths:
        print(file)
        l = file.split('/')[-1]
        f = l.split('.')[-2]
        img = Image.open(file)
        a_list = []
        for i in range(2):
            while(True):
                a = np.random.randint(1, 6)
                if a not in a_list:break
            a_list.append(a)
            print(a, f)
#        img.save(save_dir + f +'_1.tif')
#img =np.asarray(img)
            if a == 1 :
                ng2 = img.transpose(Image.FLIP_TOP_BOTTOM)
                ng2.save(save_dir +'/' + f + '_ud.tif')
            elif a == 2:
                ng3 = img.transpose(Image.FLIP_LEFT_RIGHT)
                ng3.save(save_dir +'/' + f + '_lr.tif')
            elif a == 3:
                ng4 = img.transpose(Image.ROTATE_90)
                ng4.save(save_dir +'/' + f + '_90.tif')
            elif a == 4:
                ng5 = img.transpose(Image.ROTATE_180)
                ng5.save(save_dir +'/' + f + '_180.tif')
            elif a == 5:
                ng6 = img.transpose(Image.ROTATE_270)
                ng6.save(save_dir +'/' + f + '_270.tif')
            
    return
#dir_1 ='/cptjack/totem/yatong/all_data/bach_augment_data/3/_5'
##get_transform(path_0, save_dir_0)
##get_transform(path_1, save_dir_1)
##dir_1 = '/cptjack/totem/yatong/all_data/bach_augment_data/0/_1'
##dir_5 = '/cptjack/totem/yatong/all_data/bach_augment_data/0/_5'
#save_dir = '/cptjack/totem/yatong/all_data/bach_augment_data/3'
##paths_1 = list(paths.list_images(dir_1))       
#paths_1 = list(paths.list_images(dir_1)) 
##get_transform(paths_1, save_dir)
#get_transform(paths_1,save_dir)

#save_dir = '/cptjack/totem/yatong/all_data/bach_test_augment'
save_dir = '/cptjack/totem/yatong/all_data/new_hsv_augment_data/3'
#test_dir = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Test/Photos'
#test_paths = list(paths.list_images(test_dir))
#get_transform(test_paths, save_dir)

data_dir = '/cptjack/totem/yatong/all_data/new_hsv_augment_data/3'


bach_data_0 = '/cptjack/totem/yatong/new_data/lstm_data/validation/0'
bach_data_1 = '/cptjack/totem/yatong/new_data/lstm_data/validation/1'
bach_data_2 = '/cptjack/totem/yatong/new_data/lstm_data/validation/2'
bach_data_3 = '/cptjack/totem/yatong/new_data/lstm_data/validation/3'
bach_data_paths_0 = list(paths.list_images(bach_data_0))
bach_data_paths_1 = list(paths.list_images(bach_data_1))
bach_data_paths_2 = list(paths.list_images(bach_data_2))
bach_data_paths_3 = list(paths.list_images(bach_data_3))
save_dir0 = '/cptjack/totem/yatong/all_data/new_hsv_augment_data/0'
save_dir1 = '/cptjack/totem/yatong/all_data/new_hsv_augment_data/1'
save_dir2 = '/cptjack/totem/yatong/all_data/new_hsv_augment_data/2'
save_dir3 = '/cptjack/totem/yatong/all_data/new_hsv_augment_data/3'
get_transform(bach_data_paths_0, save_dir0)
#get_transform(bach_data_paths_1, save_dir1)
#get_transform(bach_data_paths_2, save_dir2)
#get_transform(bach_data_paths_3, save_dir3)



