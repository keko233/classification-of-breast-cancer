# -*- coding: utf-8 -*-
from keras.models import load_model
import mil_data4 as d
from utils1 import generators

from keras.callbacks import CSVLogger, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import Adagrad, SGD
from keras.utils import to_categorical
from keras.applications import resnet50
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import regularizers
from keras.utils.vis_utils import plot_model
from keras.models import Sequential

from keras.layers.convolutional import  MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Flatten, Dropout 
import random
import tensorflow as tf
import time
#from generators import DataGenerator




os.environ["CUDA_VISIBLE_DEVICES"] = '2'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model_path ='/cptjack/totem/yatong/4_classes/mil_resnet50_0807/5_resnet50(224).h5'
model = load_model(model_path)
model.summary()
        
for layer in model.layers:
    layer.trainable = True

d.get_top(model, d.data_dir3_list, 3, d.result_dir, d.save_mil_dir_3, 6) 