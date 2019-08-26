# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import openslide as opsl
from PIL import Image, ImageDraw

import cv2
import os
import numpy as np

import matplotlib.pyplot as plt
from skimage import io

def create_colormap(svs_im, matrix_0, title, output_dir):
    plt_size = (svs_im.size[0] // 100, svs_im.size[1] //100)
    flg, ax = plt.subplots(figsize = plt_size, dpi =100)
    matrix = matrix_0.copy()
    matrix = cv2.resize(matrix, svs_im.size, interpolation = cv2.INTER_AREA)
    cax = ax.imshow(matrix, cmap = plt.cm.jet, alpha = 0.45)  
    svs_im_npy = np.array(svs_im.convert('RGBA'))  
    svs_im_npy[:,:][matrix[:,:] > 0] = 0  
    ax.imshow(svs_im_npy) 
    max_matrix_value = matrix.max() 
    plt.colorbar(cax, ticks = np.linspace(0, max_matrix_value, 25, endpoint = True)) 
    ax.set_title(title, fontsize = 20)
    plt.axis('off')
    
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, title ))
    plt.close('all')
    return
result_dir = '/cptjack/totem/yatong/4_classes/cnn_result/4_classes_predict_result/heatmap/'
file = '/cptjack/totem/yatong/4_classes/cnn_result/4_classes_predict_result/result_map/A10resnet50(0718)_cnn(0722)_map.png'
matrix = io.imread(file)
print(matrix.shape)
#plt_size = matrix.shape[0] //100, matrix.shape[1]//100
#flg,ax = plt.subplots(figsize = plt_size, dpi = 100)
#a = ax.imshow(matrix, cmap = plt.cm.jet)
#max_matrix_value = matrix.max()
#plt.colorbar(a, ticks = np.linspace(0, max_matrix_value, 1, endpoint = True))
#io.imsave(result_dir+'a.png',a)
#plt.imshow(a)
plt.imshow(matrix, cmap = plt.cm.jet)

plt.savefig(result_dir + 'a.png')

