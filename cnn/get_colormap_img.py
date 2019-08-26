# -*- coding: utf-8 -*-


import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

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

#out_img_dir = '/cptjack/totem/yatong/breast_predict/result/result_map'
#preview_img_dir = '/cptjack/totem/yatong/breast_predict/preview'
#preview_img_files = os.listdir(preview_img_dir)
##out_img_files = os.listdir(out_img_path)
#colormap_dir = '/cptjack/totem/yatong/breast_predict/result/colormap'

#for filename in preview_img_files:
#    name = filename.split('_')
#    out_img_filename = name[0] + '_' + name[1] + '_' + 'map.png'
#    title = name[0] + '_' + name[1] + '_' + 'colormap'
#    preview_img_path = os.path.sep.join([preview_img_dir, filename])
#    out_img_path = os.path.sep.join([out_img_dir, out_img_filename])
#    preview_file = Image.open(preview_img_path)
#    out_img_file = matplotlib.image.imread(out_img_path)
#   # print(out_img_file)
#    create_colormap(preview_file, out_img_file, title,  colormap_dir)
















