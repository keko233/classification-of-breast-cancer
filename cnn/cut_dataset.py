# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
from skimage import io


'''
无重叠地将2048x1536的大图切割为512x512的patch
file：要切割的单张大图保存路径
save_dir:切图得到的patch的保存路径
'''
def cut_img(file, save_dir):
#    l = file.split('/')[-2]
#    print(l)
    f = file.split('/')[-1]
    f_name = f.split('.')[-2]
    img = Image.open(file)
    img = np.asarray(img)
    step = 512
    h_count = img.shape[0] // step
    w_count = img.shape[1] // step
    i = 0 
    print(f_name,h_count, w_count)
    
    for y in range(h_count):
        for x in range(0,w_count):
            x0 =  x * step
            x1 = x0 + step
            y0 = y * step
            y1 = y0 + step
#            print(x0,x1,y0,y1)
            patch = img[y0:y1, x0:x1]
            rgb_s = (abs(patch[:,:,0] -107) >= 93) & (abs(patch[:,:,1] -107) >= 93) & (abs(patch[:,:,2] -107) >= 93)
            if np.sum(rgb_s)<=(step * step ) * 0.4:
                i = i + 1
                io.imsave(save_dir + '/' + f_name + '_'+str(i) +'.png', patch)
    return
    
def get_dataset(paths,save_dir):
    for file in paths:
        cut_img(file,save_dir)
        



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    