#-*- coding: utf-8 -*-
'''
更改bach challenge PartB 提交图片的大小
必须与官网要求的一致才能有分数
'''
from skimage import io
import cv2
import os

'''
file_dir:存储原始partB 结果图的路径
save_dir :存储改变大小后的结果图的路径
'''
def change_img(file_dir, save_dir):
    #官网上所有svs图对应结果图的大小。其对应命名图片是test1 到test10。
    pixels_list = [(10663, 9398), (13782, 11223), (8798, 8344), (15738, 8531), (14262, 9062), (12711, 6993), (14096, 10396), (9995, 7497), (12906, 10935), (14483, 10928)]
    
    files = os.listdir(file_dir)
    #按照1到10的图片名字顺序将对应的svs图片排序
    files.sort(key = lambda a:int(a.split('_')[0][4:6]))
    print(files)
    for index, file in enumerate(files):
        index2 = index + 1
        file_paths = os.path.sep.join([file_dir, file])
        print(index, file_paths)
        print(pixels_list[index])
        img = io.imread(file_paths)
        out_img = cv2.resize(img, (pixels_list[index]), interpolation=cv2.INTER_AREA)
        io.imsave(save_dir + str(index2) + '.png' ,out_img)
    return

file_dir = '/cptjack/totem/yatong/bach_svs_result/resnet50_newhsv_0813/result_map/map'
save_dir = '/cptjack/totem/yatong/bach_svs_result/temp/'