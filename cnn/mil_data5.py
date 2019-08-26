# -*- coding: utf-8 -*-
"""
2019-08-12  yatong
获取用于多示例学习的训练集
每个bag（即2048x1536的大图）中取三张512x512的patch加入训练集中
"""
import os
from imutils import paths
import numpy as np
import cv2
from skimage import io
from PIL import Image


data_dir = '/cptjack/totem/yatong/all_data/new_hsv_augment_data/'
data_dir0 = os.path.sep.join([data_dir, '0'])
data_dir1 = os.path.sep.join([data_dir, '1'])
data_dir2 = os.path.sep.join([data_dir, '2'])
data_dir3 = os.path.sep.join([data_dir, '3'])

data_dir0_list = list(paths.list_images(data_dir0))
data_dir1_list = list(paths.list_images(data_dir1))
data_dir2_list = list(paths.list_images(data_dir2))
data_dir3_list = list(paths.list_images(data_dir3))

save_mil_dir_0 = '/cptjack/totem/yatong/all_data/mil_new_512_2/train/0'
save_mil_dir_1 = '/cptjack/totem/yatong/all_data/mil_new_512_2/train/1'
save_mil_dir_2 = '/cptjack/totem/yatong/all_data/mil_new_512_2/train/2'
save_mil_dir_3 = '/cptjack/totem/yatong/all_data/mil_new_512_2/train/3'

result_dir = '/cptjack/totem/yatong/4_classes/picture0813'

'''
把切割的512x512的patch拼接起来
'''
def combine_images(img, w,h):
    shape = img.shape[1:4]
    image = np.zeros((h * shape[0], w *shape[1], shape[2]),
                     dtype = img.dtype)
    for index, img1 in enumerate(img):
        i = int(index/w)
        j = index % w
        image[i * shape[0]:(i+1)*shape[0], j * shape[1]:(j+1)*shape[1],:] = img1[:,:,:]
    return image

'''
data_dir_list:输入图片的保存路径列表
label：输入图片的标签
result_dir：保存可视化图片的路径
save_mil_dir:保存选取图片的路径
t：第几轮挑选的训练集
flag：flag为True时挑选概率最高的前三张patch，为Flase时挑选概率最低的前三张patch
'''
def get_top(model, data_dir_list, label, result_dir, save_mil_dir, t, flag):  
    d = 'epoch_' + str(t)
    result_dir2 = os.path.sep.join([result_dir, d])
    add_dir = os.path.sep.join([result_dir2, 'add_img'])
    out_img_dir = os.path.sep.join([result_dir2, 'out_img'])
    combine_dir = os.path.sep.join([result_dir2, 'combine_img'])
    if not os.path.exists(out_img_dir):os.makedirs(out_img_dir)
    if not os.path.exists(add_dir):os.makedirs(add_dir)
    if not os.path.exists(combine_dir):os.makedirs(combine_dir)   
    for p in data_dir_list:
        print(p)
        f = p.split('/')[-1]
        n = f.split('.')[-2]
        img = io.imread(p)
        img = np.uint8(img)
        step = 512
        step2 = 256
#        name_list = []
        coordinate_list = []
        h_count = ((img.shape[0] - step) // step2) + 1
        w_count = ((img.shape[1] - step) // step2) + 1
        images = np.zeros([0, 224,224,3])
        out_img = np.zeros([h_count, w_count,3])
#        title = n + '_overlap_colormap'
        prob_list = []
        i = 0
        for y in range(h_count):
            for x in range(w_count):
                i = i+1
                patch = img[y * step2:(y * step2)+ step, x * step2:(x * step2) + step]
                img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
                img2 = img1.reshape(1,224,224,3)
                output = model.predict(img2/255.0)
                prob_list.append(output[0][label])
                coordinate_list.append((y,x))
                pro = round(output[0][label], 4)
                font = cv2.FONT_HERSHEY_SIMPLEX
                img3 = cv2.putText(img1, str(pro), (1,112), font, 1, (0,0,0), 2)
                img3 = img3.reshape(1,224,224,3)
                images = np.concatenate([images, img3], axis = 0)
       
        overlap_img = combine_images(images, w_count, h_count)
        io.imsave(combine_dir + '/' + n + '.png', overlap_img/255.0)
#        data = list(zip(prob_list, name_list, coordinate_list))
        data = list(zip(prob_list, coordinate_list))
        data.sort(reverse = flag)
#        prob_list, name_list, coordinate_list = zip(*data)
        prob_list, coordinate_list = zip(*data)
        coordinate_list = list(coordinate_list)
        top_coordinate = coordinate_list[0:3]
        for index,c in enumerate(top_coordinate):
            y = c[0]
            x = c[1]
            patch = img[y * step2:(y * step2)+ step, x * step2:(x * step2) + step]
            rgb_s1 = (abs(patch[:,:,0] -107) >= 93) & (abs(patch[:,:,1] -107) >= 93) & (abs(patch[:,:,2] -107) >= 93)
            if np.sum(rgb_s1)<=(step * step ) * 0.45:
                io.imsave(save_mil_dir + '/'+ n+ '_'+str(index) + '.png', patch)
                out_img[y,x,0] =int(255)
                out_img[y,x,1] =int(255)
                out_img[y,x,2] =int(255)
        out_img = cv2.resize(out_img, (overlap_img.shape[1], overlap_img.shape[0]), interpolation = cv2.INTER_AREA)
        io.imsave(out_img_dir + '/' + n +'.png', out_img/255.0)
        out_img2 = Image.open(out_img_dir + '/' + n +'.png')
        overlap_img2 = Image.open(combine_dir + '/' + n + '.png')
        add_img = Image.blend(overlap_img2, out_img2, 0.3)
        io.imsave(add_dir + '/' + n +'.png', add_img)
    return
