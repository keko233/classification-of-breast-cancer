# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.callbacks import CSVLogger, EarlyStopping
from keras.callbacks import ModelCheckpoint,  ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np
import os
import time
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

timesteps = 9
num_classes = 2
bs = 32
num_epochs = 100
init_lr = 1e-3

save_dir ='/cptjack/totem/yatong/4_classes/lstm_data'


train_data = np.load(save_dir + '/train_data(inceptionResnetV2_0806_3).npy')
train_labels = np.load(save_dir + '/train_labels(inceptionResnetV2_0806_3).npy')

#将训练集的标签进行ong-hot编码
train_labels = to_categorical(train_labels)

trainTotals = len(train_data)
#打乱训练集
train = list(zip(train_data, train_labels))
print(len(train))
random.seed(62)
random.shuffle(train)
train_data, train_labels = zip(*train)

val_data = np.load(save_dir + '/val_data(inceptionResnetV2_0806_3).npy')
val_labels = np.load(save_dir + '/val_labels(inceptionResnetV2_0806_3).npy')
val_labels = to_categorical(val_labels)
val = (val_data, val_labels)
valTotals = len(val_data)


train_data = np.array(train_data) 
train_labels = np.array(train_labels)
val_data = np.array(val_data)
val_labels = np.array(val_labels)

classTotals = train_labels.sum(axis = 0)

#获得各个类的比例
classWeight = classTotals.max() / classTotals
print(classWeight)



lstm_model = Sequential()
lstm_model.add(Bidirectional(LSTM(256, return_sequences = True),input_shape = (train_data.shape[1], train_data.shape[2])))
lstm_model.add(Bidirectional(LSTM(256,return_sequences = True)))
lstm_model.add(Bidirectional(LSTM(256,return_sequences = True)))
lstm_model.add(Bidirectional(LSTM(256,return_sequences = False)))
lstm_model.add(Dense(4, activation = 'softmax'))
lstm_model.summary()


opt = SGD(lr=init_lr, decay=init_lr/num_epochs, 
                     momentum=0.9, nesterov=True)

lstm_model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"]
              )
#import get_lstm_train_data as gltd
#import numpy as np

#def generator_batch_data_random(x, y, batch_size):
#    ylen = len(y)
#    loop_count = ylen // batch_size
#    while(True):
#        i = random.randint(0, loop_count)
#        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) *batch_size]


class Mycbk(ModelCheckpoint):
    def __init__(self, model, filepath ,monitor = 'val_acc',mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
    def set_model(self,model):
        super(Mycbk,self).set_model(self.single_model)

def get_callbacks(filepath,model,patience):
    es = EarlyStopping('val_acc', patience=patience, mode="min")
    msave = Mycbk(model,'./inceptionResnetV2_0806_3_lstm(0807)/'+ filepath + '.h5') 
    file_dir = './inceptionResnetV2_0806_3_lstm(0807)/log/'+str(filepath) + '/' + time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, 
                                  patience=2, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger('./inceptionResnetV2_0806_3_lstm(0807)/' + time.strftime('%Y_%m_%d',time.localtime(time.time())) +'_'+str(filepath) +'_log.csv', separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]

file = 'lstm_2'
callbacks_s = get_callbacks(file,lstm_model,patience=50)

#一次性读取所有数据训练
H = lstm_model.fit(train_data, train_labels,
              epochs = num_epochs,
              validation_data = val,
              class_weight = classWeight,
              steps_per_epoch = trainTotals // bs,
              validation_steps = valTotals // bs,
              callbacks = callbacks_s, verbose = 1)

#H = lstm_model.fit_generator(generator_batch_data_random(train_data,train_labels , bs),
#              epochs = num_epochs,
#              validation_data = val,
#              class_weight = classWeight,
#              steps_per_epoch = trainTotals // bs,
#              validation_steps = valTotals // bs,
#              callbacks = callbacks_s, verbose = 1)














