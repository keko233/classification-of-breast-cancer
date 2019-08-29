# -*- coding: utf-8 -*-
from PIL import Image


'''
Paths：需要进行数据增强的数据的保存路径
save_dir:数据增强后的图片保存路径
一张图片生成5张增强后的图片，分别为：上下翻转，左右翻转，90度旋转，180度旋转，270度旋转
'''
def get_transform(Paths, save_dir):
    for file in Paths:
        print(file)
        l = file.split('/')[-1]
        f = l.split('.')[-2]
        img = Image.open(file)
        ng2 = img.transpose(Image.FLIP_TOP_BOTTOM)
        ng2.save(save_dir +'/' + f + '_ud.tif')
        ng3 = img.transpose(Image.FLIP_LEFT_RIGHT)
        ng3.save(save_dir +'/' + f + '_lr.tif')
        ng4 = img.transpose(Image.ROTATE_90)
        ng4.save(save_dir +'/' + f + '_90.tif')
        ng5 = img.transpose(Image.ROTATE_180)
        ng5.save(save_dir +'/' + f + '_180.tif')
        ng6 = img.transpose(Image.ROTATE_270)
        ng6.save(save_dir +'/' + f + '_270.tif')         
    return




