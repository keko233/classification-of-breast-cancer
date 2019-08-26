# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from imutils import paths




base_dir1 = '/cptjack/totem/breast_part1/breast_cancer_pathological_image'
base_dir2 = '/cptjack/totem/Breast_part2/methods--resize--20190123'

#bach_data = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/Photos'
bach_data1 = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/Photos/Benign'
bach_data2 = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/Photos/Insitu'
bach_data3 = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/Photos/Normal'
bach_data4 = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/Photos/Invasive'

bach_data = [bach_data1, bach_data2, bach_data3, bach_data4]
def set_dir(base_dir):
    n = []
    name_list = ['Benign/Benign 40x', 'In Situ/In Situ 40x', 'Normal/Normal 40x', 'Invasive/Invasive 40x']
    for name in name_list:
        p = os.path.sep.join([base_dir, name])

        print(p)
        n.append(p)
    return n

base_dir1 = set_dir(base_dir1)
base_dir2 = set_dir(base_dir2)
#bach_data = set_dir(bach_data)
print(base_dir1)

def get_dataset(name):
    normal_class0_paths = list(paths.list_images(name[2]))
    benign_class1_paths = list(paths.list_images(name[0]))
    insitu_class2_paths = list(paths.list_images(name[1]))
    invasive_class3_paths = list(paths.list_images(name[3]))
    
    return normal_class0_paths, benign_class1_paths, insitu_class2_paths, invasive_class3_paths

normal_class0_paths1, benign_class1_paths1, insitu_class2_paths1, invasive_class3_paths1 = get_dataset(base_dir1)   
normal_class0_paths2, benign_class1_paths2, insitu_class2_paths2, invasive_class3_paths2 = get_dataset(base_dir2) 
train_normal_class0_paths = normal_class0_paths1 + normal_class0_paths2
train_benign_class1_paths = benign_class1_paths1 + benign_class1_paths2
train_insitu_class2_paths = insitu_class2_paths1 + insitu_class2_paths2
train_invasive_class3_paths = invasive_class3_paths1 + invasive_class3_paths2
print(len(train_normal_class0_paths), len(train_benign_class1_paths), len(train_insitu_class2_paths),len(train_invasive_class3_paths))
                  
val_normal_class0_paths, val_benign_class1_paths, val_insitu_class2_paths, val_invasive_class3_paths = get_dataset(bach_data)
print(len(val_normal_class0_paths), len(val_benign_class1_paths), len(val_insitu_class2_paths), len(val_invasive_class3_paths))
