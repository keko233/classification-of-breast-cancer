# -*- coding: utf-8 -*-
'''
下载并保存有imagenet权重的efficientne
'''
from keras_efficientnets import EfficientNetB4
import os 
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
model = EfficientNetB4(input_shape = (224,224,3),classes = 2, include_top = False, weights = 'imagenet')
model.save('/cptjack/sys_software_bak/tensorflow_keras_models/models/EfficientNetB4(224).h5')
