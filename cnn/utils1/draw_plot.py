# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
#p = '/cptjack/totem/yatong/new_data/inceptionv3_3/2019_07_03_<keras.models.Sequential object at 0x7ffa1c0b5128>_log.csv'
#p ='/cptjack/totem/yatong/new_data/EfficientNet_1/2019_07_04_<keras.engine.sequential.Sequential object at 0x7fbfd2178e80>_log.csv'
#p = '/cptjack/totem/yatong/new_data/lstm_1(0704)/2019_07_04_<keras.models.Sequential object at 0x7f61e4d65780>_log.csv'
#p = '/cptjack/totem/yatong/new_data/lstm_2(0704)/2019_07_04_<keras.engine.sequential.Sequential object at 0x7f89600f94a8>_log.csv'
#p = '/cptjack/totem/yatong/new_data/lstm_3(0705)/2019_07_05_<keras.engine.sequential.Sequential object at 0x7f27755b95c0>_log.csv'
#p = '/cptjack/totem/yatong/new_data/inceptionResnetV2/2019_07_11_<keras.engine.sequential.Sequential object at 0x7f01210d3470>_log.csv'
#p = '/cptjack/totem/yatong/new_data/EfficientNetB3_2/2019_07_12_<keras.engine.sequential.Sequential object at 0x7fbb308fcc18>_log.csv'
#p = '/cptjack/totem/yatong/new_data/inceptionResnetV2(0716)/2019_07_16_<keras.engine.sequential.Sequential object at 0x7f9072e78ef0>_log.csv'
#p = '/cptjack/totem/yatong/new_data/inceptionResnetV2(0718)/2019_07_18_<keras.engine.sequential.Sequential object at 0x7f954e938f28>_log.csv'
#p = '/cptjack/totem/yatong/4_classes/resnet50_0718/2019_07_18_<keras.engine.sequential.Sequential object at 0x7fd313b01d68>_log.csv'
#p = '/cptjack/totem/yatong/4_classes/resnet50_0724/2019_07_24_<keras.engine.sequential.Sequential object at 0x7ff07ed13fd0>_log.csv'
p = '/cptjack/totem/yatong/4_classes/resnet50_0725/2019_07_25_<keras.engine.sequential.Sequential object at 0x7f7d5bf6fdd8>_log.csv'
data = pd.read_csv(p)
acc = data['acc']
loss = data['loss']
val_loss = data['val_loss']
val_acc = data['val_acc']
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(loss, color = 'r', linestyle = 'dashed', marker = 'x', label = 'loss')
plt.plot(val_loss, color = 'k', linestyle = 'dashed',marker = 'x', label = 'val_loss')
plt.plot(acc, color = 'r', linestyle = 'dashed', marker = 'o', label = 'acc')
plt.plot(val_acc, color = 'k', linestyle = 'dashed',marker = 'o', label = 'val_acc')
plt.legend(loc = 'best')
ax.set_title('')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss/Accuracy')

#n = 18
#plt.style.use("ggplot")
#plt.figure()
#plt.plot(np.arange(0, n), loss , label="train_loss")
#plt.plot(np.arange(0, n), val_loss, label="val_loss")
#plt.plot(np.arange(0, n), acc, label="train_acc")
#plt.plot(np.arange(0, n), val_acc, label="val_acc")
#plt.title("Training Loss and Accuracy on Dataset")
#plt.xlabel("Epoch")
#plt.ylabel("Loss/Accuracy")
#plt.legend(loc="lower left")
plt.savefig(args["plot"])