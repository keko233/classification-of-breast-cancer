# -*- coding: utf-8 -*-
'''
从keras上加载权重并保存到文件夹
'''
#from keras.applications import resnet50
from keras.applications import inception_resnet_v2
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#model = resnet50.ResNet50(include_top = None, weights = 'imagenet', 
#                               input_shape = (224,224,3))
#
#model.summary()
#model.save('/cptjack/sys_software_bak/tensorflow_keras_models/models/resnet50(224).h5')

model = inception_resnet_v2.InceptionResNetV2(include_top = None, weights = 'imagenet', 
                               input_shape = (224,224,3))

model.summary()
model.save('/cptjack/sys_software_bak/tensorflow_keras_models/models/InceptionResNet(224).h5')