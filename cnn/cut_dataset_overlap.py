# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
from skimage import io


def cut_img(file, save_dir):

    f = file.split('/')[-1]
    f_name = f.split('.')[-2]
    img = Image.open(file)
    img = np.asarray(img)
    step = 512
    step2 = 256
#    print(img.shape)

    h_count = ((img.shape[0] - step) // step2 ) + 1
    w_count = ((img.shape[1] - step) // step2 ) + 1
    i = 0 
#    print(f_name,h_count, w_count)
    
    for y in range(h_count):
        for x in range(0,w_count):
            x0 =  x * step2
            x1 = x0 + step
            y0 = y * step2
            y1 = y0 + step
#            print(x0,x1,y0,y1)
            patch = img[y0:y1, x0:x1]
            rgb_s = (abs(patch[:,:,0] -107) >= 93) & (abs(patch[:,:,1] -107) >= 93) & (abs(patch[:,:,2] -107) >= 93)
            if np.sum(rgb_s)<=(step * step ) * 0.4:
                i = i + 1
                io.imsave(save_dir + '/' + f_name + '_'+str(i) +'.png', patch)
    return

'''
Paths:包含每一个图片路径的列表
save_dir：保存处理后图片的路径
'''
def get_dataset(Paths,save_dir):
    for file in Paths:
        cut_img(file, save_dir)
    return



