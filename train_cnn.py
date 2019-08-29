# -*- coding: utf-8 -*-


from keras import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import CSVLogger,  EarlyStopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras.layers.core import Dense, Flatten, Dropout 
from keras import regularizers

from imutils import paths
import time
import os




os.environ["CUDA_VISIBLE_DEVICES"] = '1'


file_path = '/cptjack/sys_software_bak/tensorflow_keras_models/models/resnet50(224).h5'
#file_path = '/cptjack/sys_software_bak/tensorflow_keras_models/models/InceptionResnetV2(224).h5'

base_model = load_model(file_path)
base_model.summary()
model = Sequential()
model.add(base_model)
model.add(Flatten())

model.add(Dense(32,activation='relu',kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.summary()


for layer in model.layers:
    layer.trainable = True

#超参数设置    
num_epochs = 30
init_lr = 1e-2
bs = 32

#训练集以及验证集路径
val_dir = '/cptjack/totem/yatong/all_data/val'
train_dir = '/cptjack/totem/yatong/all_data/new_hsv_augment_data_512/train'

opt = SGD(lr=init_lr, decay=init_lr/num_epochs, 
                     momentum=0.9, nesterov=True)


model.compile(loss = "weight_categorical_crossentropy", optimizer = opt,
                  metrics = ["accuracy"])
#top_model.compile(loss="categorical_crossentropy", optimizer=opt,
#              metrics=["accuracy"]
#              )

trainPaths = list(paths.list_images(train_dir))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(val_dir)))

trainAug = ImageDataGenerator(rescale=1/255.0,
                              horizontal_flip=True,
                              vertical_flip=True,
                              zoom_range = 0.2)

valAug = ImageDataGenerator(rescale=1/255.0)

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

#testGen = valAug.flow_from_directory(config.test_dir,
#                                        class_mode="categorical",
#                                        target_size=(224,224),
#                                        color_mode="rgb",
#                                        shuffle=False,
#                                        batch_size=bs)


class Mycbk(ModelCheckpoint):
    def __init__(self, model, filepath ,monitor = 'val_acc',mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
    def set_model(self,model):
        super(Mycbk,self).set_model(self.single_model)

def get_callbacks(filepath,model,patience):
    es = EarlyStopping('val_acc', patience=patience, mode="min")
    msave = Mycbk(model,'./resnet50_newhsv_0813/'+ filepath + '.h5') 
    file_dir = './resnet50_newhsv_0813/log/'+str(filepath) + '/' + time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, 
                                  patience=2, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger('./resnet50_newhsv_0813/' + time.strftime('%Y_%m_%d',time.localtime(time.time())) +'_'+str(filepath) +'_log.csv', separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]

#file = 'InceptionResnetV2(224)'
file = 'resnet50'
callbacks_s = get_callbacks(file,model,patience=10)
model.fit_generator(trainGen,
                        steps_per_epoch=totalTrain // bs,
                        validation_data=valGen,
                        validation_steps=totalVal // bs,
#                        class_weight=classWeight,
                        epochs=num_epochs,
                        callbacks=callbacks_s,
                        verbose=1)








