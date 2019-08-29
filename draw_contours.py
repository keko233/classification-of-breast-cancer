# -*- coding: utf-8 -*-
'''
画出svs图中被预测为invasive的区域轮廓
'''
import cv2
import numpy as np
from skimage import io   
import DelaunayFill as de

#结果矩阵保存路径（保存的结果为svs图中每个patch预测为invasive的概率再乘以255的值)
result_img_path = '/cptjack/totem/yatong/4_classes/lstm_result/resnet50(0725)_lstm/result_map/A04_32(0730)_map.png'


#画好标注的svs二级缩略图保存路径
preview_path = '/cptjack/totem/yatong/bach_challenge/result/preview/A04.png'

#生成图片的保存路径
result_path = '/cptjack/totem/yatong/picture_processing/result/0801/'

#概率的阈值。即预测为invasive的概率超过这个阈值就判断这个patch为invasive
threshod = 0.5 
#threshod = 0.5 * 255

#读取结果矩阵
img = io.imread(result_img_path)

#读取标注图
pre_img = io.imread(preview_path)
#img = np.uint8(img)
print(img.shape)
w_count = img.shape[0]
h_count = img.shape[1]

#将预测为invasive的值都设为255，得到新的结果矩阵
out = np.zeros([w_count, h_count])
for x in range(w_count):
    for y in range(h_count):
        if (img[x][y] / 255) >= threshod:
#            print(img[x][y] / 255)
            out[x][y] = 255
        else:
#            print(img[x][y] / 255)
            out[x][y] = 0


print(out)
print(sum(sum(out)))
out = np.uint8(out)
io.imshow(out)
io.imsave(result_path +'a02out.png', out)
#设置卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
#img1 = cv2.dilate(out,kernel)
#io.imshow(img1)
#img1 = cv2.erode(img1, kernel)
#io.imshow(img1)
#img1 = cv2.dilate(img1, kernel)
#io.imshow(img1)
#img1 = cv2.erode(img1, kernel)
#io.imshow(img1)

#对新的结果矩阵进行开运算
img1 = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
io.imshow(img1)
#io.imsave(result_path +'a02_morp.png',img1)
#plt.imshow(img1)

#通过三角剖分算法得到填充后的区域
triangle = de.get_triangle_matrix(img1,300,60)
io.imshow(triangle)
io.imsave(result_path + 'a02_tri.png', triangle)
a = io.imread(result_path + 'a02_tri.png')

#找轮廓
d,e,f = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(e):
    #print(cv2.contourArea(e[i]))
    if cv2.contourArea(e[i]) > 1000:
        print('get')
        print(cv2.contourArea(e[i]))
        cv2.drawContours(pre_img, e, i, (76, 177, 34), 10)
io.imshow(pre_img)
io.imsave(result_path + 'pre_img_A02.png', pre_img)
