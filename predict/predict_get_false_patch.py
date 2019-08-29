# -*- coding: utf-8 -*-
#from keras.models import load_model
#from imutils import paths
import numpy as np
import cv2
import os
from skimage import io
#import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#gpu_options=tf.GPUOptions(allow_growth=True)
#sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
#model_path = '/cptjack/totem/yatong/4_classes/resnet50_0718/resnet50(224).h5'      
#model = load_model(model_path)
'''
输入512x512的patch进行预测，将预测为不是该patch对应标签的patch保存起来，即得到预测错误的样本
model：用于预测的模型
Paths：用于预测的每个patch的路径列表
save_f_dir：用于保存预测错误样本的路径

返回值：name：保存错误样本命名的列表
'''
def get_false_patch(model,Paths,save_f_dir):
    print(len(Paths))
    name = []
    for p in Paths:
# i是根据文件命名得到的这个patch对应的标签
        i = p.split('/')[-2]
        f = p.split('/')[-1]
        n = f.split('.')[-2]
        print(f)
        img = io.imread(p)
        img1 = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        img1 = img1.reshape(1,224,224,3)
        output = model.predict(img1/255.0)
        preIndex = np.argmax(output, axis = 1)
        if preIndex[0] != int(i) :
            save_dir = os.path.sep.join([save_f_dir, str(preIndex[0])])
            if not os.path.exists(save_dir):os.makedirs(save_dir)
            name.append(f)
            print(preIndex[0], i)
            io.imsave(save_dir +'/'+ n +'.png', img)
    return name


     
#save_base_dir = '/cptjack/totem/yatong/all_data/balance_normalized_dataset_512'
#train_dir = os.path.sep.join([save_base_dir, 'train'])
#val_dir = os.path.sep.join([save_base_dir, 'validation'])
#
#val_class0_dir = os.path.sep.join([val_dir, '0'])
#val_class1_dir = os.path.sep.join([val_dir, '1'])
#val_class2_dir = os.path.sep.join([val_dir, '2'])
#val_class3_dir = os.path.sep.join([val_dir, '3'])
#
#data_paths_0 = list(paths.list_images(val_class0_dir))
#data_paths_1 = list(paths.list_images(val_class1_dir))
#data_paths_2 = list(paths.list_images(val_class2_dir))
#data_paths_3 = list(paths.list_images(val_class3_dir))
#
#save_f_dir = '/cptjack/totem/yatong/4_classes/resnet50_0718_predict_stain_normalized_f'
#save_f_0_dir = os.path.sep.join([save_f_dir, '0'])
#save_f_1_dir = os.path.sep.join([save_f_dir, '1'])
#save_f_2_dir = os.path.sep.join([save_f_dir, '2'])
#save_f_3_dir = os.path.sep.join([save_f_dir, '3'])





