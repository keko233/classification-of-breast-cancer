# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
from skimage import io
#import cv2
import os
from imutils import paths
import random
import shutil


#save_base_dir = '/cptjack/totem/yatong/new_data/dataset'
#save_base_dir = '/cptjack/totem/yatong/all_data/balance_normalized_dataset_512'
save_base_dir = '/cptjack/totem/yatong/all_data/bach_augment_data_512'
train_dir = os.path.sep.join([save_base_dir, 'train'])
val_dir = os.path.sep.join([save_base_dir, 'validation'])

train_class0_dir = os.path.sep.join([train_dir, '0'])
train_class1_dir = os.path.sep.join([train_dir, '1'])
train_class2_dir = os.path.sep.join([train_dir, '2'])
train_class3_dir = os.path.sep.join([train_dir, '3'])

val_class0_dir = os.path.sep.join([val_dir, '0'])
val_class1_dir = os.path.sep.join([val_dir, '1'])
val_class2_dir = os.path.sep.join([val_dir, '2'])
val_class3_dir = os.path.sep.join([val_dir, '3'])

val_class0 = list(paths.list_images(val_class0_dir))
val_class1 = list(paths.list_images(val_class1_dir))
val_class2 = list(paths.list_images(val_class2_dir))
val_class3 = list(paths.list_images(val_class3_dir))

random.seed(34)
random.shuffle(val_class0)
random.shuffle(val_class1)
random.shuffle(val_class2)
random.shuffle(val_class3)
d0 = val_class0[0:200]
d1 = val_class1[0:200]
d2 = val_class2[0:200]
d3 = val_class3[0:200]

result_dir = '/cptjack/totem/yatong/all_data/val'
r0 = os.path.sep.join([result_dir ,'0'])
r1 = os.path.sep.join([result_dir ,'1'])
r2 = os.path.sep.join([result_dir ,'2'])
r3 = os.path.sep.join([result_dir ,'3'])

def c(r,d):
    for f in d:
        shutil.copy2(f, r)
    return

c(r0, d0)
c(r1, d1)
c(r2, d2)
c(r3, d3)


