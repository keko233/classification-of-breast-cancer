# -*- coding: utf-8 -*-
"""
2019-08-12 yatong
"""

from keras.models import load_model
import mil_data5 as d
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

def training(i):
    #第0轮使用imagenet权重
    if i == 0:
        file_path = '/cptjack/sys_software_bak/tensorflow_keras_models/models/resnet50(224).h5'
        #file_path = '/cptjack/sys_software_bak/tensorflow_keras_models/models/InceptionResnetV2(224).h5'
        base_model = load_model(file_path)
        base_model.summary()
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(32,activation='relu',kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(4, activation='softmax'))
        model.summary()   
        
        for layer in model.layers:
            layer.trainable = True
            
        d.get_top(model, d.data_dir0_list, 0, d.result_dir, d.save_mil_dir_0, i, False)
        d.get_top(model, d.data_dir1_list, 1, d.result_dir, d.save_mil_dir_1, i, True)
        d.get_top(model, d.data_dir2_list, 2, d.result_dir, d.save_mil_dir_2, i, True)
        d.get_top(model, d.data_dir3_list, 3, d.result_dir, d.save_mil_dir_3, i, True)   
        
        
    #使用上一轮训练得到模型来预测挑选patch       
    else:
        file_dir = '/cptjack/totem/yatong/4_classes/new_mil_resnet50_0813'
        if not os.path.exists(file_dir):os.makedirs(file_dir)
        l = i-1
        name = str(l) + '_resnet50(224).h5'
        model_path = os.path.sep.join([file_dir, name])
        print(model_path)
        print(i)
        model = load_model(model_path)
        model.summary()
        
        for layer in model.layers:
            layer.trainable = True
            
        import shutil
        shutil.rmtree(d.save_mil_dir_0)
        os.makedirs(d.save_mil_dir_0)
        shutil.rmtree(d.save_mil_dir_1)
        os.makedirs(d.save_mil_dir_1)
        shutil.rmtree(d.save_mil_dir_2)
        os.makedirs(d.save_mil_dir_2)
        shutil.rmtree(d.save_mil_dir_3)
        os.makedirs(d.save_mil_dir_3)
        
        '''
        选取新一轮的训练集
        '''
        d.get_top(model, d.data_dir0_list, 0, d.result_dir, d.save_mil_dir_0, i, False)
        d.get_top(model, d.data_dir1_list, 1, d.result_dir, d.save_mil_dir_1, i, True)
        d.get_top(model, d.data_dir2_list, 2, d.result_dir, d.save_mil_dir_2, i, True)
        d.get_top(model, d.data_dir3_list, 3, d.result_dir, d.save_mil_dir_3, i, True)  
        
    num_epochs = 3
    init_lr = 1e-2
    bs = 32
      

    
    train_dir = '/cptjack/totem/yatong/all_data/mil_new_512_2/train'
    val_dir = '/cptjack/totem/yatong/all_data/val'
    
    opt = SGD(lr=init_lr, decay=init_lr/num_epochs, 
                         momentum=0.9, nesterov=True)
    
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    
    trainPaths = list(paths.list_images(train_dir))
    random.seed(40)
    random.shuffle(trainPaths)
    totalTrain = len(trainPaths)
    totalVal = len(list(paths.list_images(val_dir)))
    
    trainAug = generators.DataGenerator(rescale=1/255.0)
    
    valAug = generators.DataGenerator(rescale=1/255.0)
    
    trainGen = trainAug.flow_from_directory(train_dir,
                                            class_mode="categorical",
                                            target_size=(224,224),
                                            color_mode="rgb",
                                            shuffle=True,
                                            batch_size=bs)
    
    
    valGen = valAug.flow_from_directory(val_dir,
                                            class_mode="categorical",
                                            target_size=(224,224),
                                            color_mode="rgb",
                                            shuffle=True,
                                            batch_size=bs)
    
    class Mycbk(ModelCheckpoint):
        def __init__(self, model, filepath ,monitor = 'val_acc',mode='min', save_best_only=True):
            self.single_model = model
            super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
        def set_model(self,model):
            super(Mycbk,self).set_model(self.single_model)
    
    def get_callbacks(i,filepath,model,patience):
        es = EarlyStopping('val_acc', patience=patience, mode="min")
        msave = Mycbk(model,'./new_mil_resnet50_0813/'+ filepath) 
        file_dir = './new_mil_resnet50_0813/log/'+str(i) + '/' + time.strftime('%Y_%m_%d',time.localtime(time.time()))
        if not os.path.exists(file_dir): os.makedirs(file_dir)
        tb_log = TensorBoard(log_dir=file_dir)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, 
                                      patience=2, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
        log_cv = CSVLogger('./new_mil_resnet50_0813/' + time.strftime('%Y_%m_%d',time.localtime(time.time())) +'_'+str(i) +'_log.csv', separator=',', append=True)
        return [es, msave,reduce_lr,tb_log,log_cv]
    
    file = 'resnet50(224).h5'
    file2 = str(i) + '_' + file
    callbacks_s = get_callbacks(i,file2,model,patience=2)
    H = model.fit_generator(trainGen,
                            steps_per_epoch=totalTrain // bs,
                            validation_data=valGen,
                            validation_steps=totalVal // bs,
                            epochs=num_epochs,
                            callbacks=callbacks_s,
                            verbose=1)
    return



'''
训练八轮，第一轮用imagenet权重预测挑选patch
每一轮都用上一轮训练保存的模型来预测得到最好的patch构成新的训练集
'''
for i in range(0,8):
    training(i)

