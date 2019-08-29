# -*- coding: utf-8 -*-
'''
lstm模型预测svs图片
'''
from __future__ import division
from utils import classes4_preview as preview_3
from utils import get_colormap_img as colormap
import openslide as opsl
import numpy as np
import cv2
import os
from keras.models import load_model
import gc
from keras.models import Model
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

'''
将1536x1536图片切割为9张512x512的patch，按顺序输入模型中得到对应的cnn特征
输入：
    img:从svs图中切出的1536x1536的图片
    layer_model：用于提取cnn特征的cnn layer model
输出：
    data：包含一张1536x1536的图片的cnn特征--> 1x9x32维的特征（32维为瓶颈特征维度，可根据模型的不同而更改）
'''
def get_imgdata(img, layer_model):
    data = []
    img = np.asarray(img)
    step = 512
    h_count = img.shape[0] // step
    w_count = img.shape[1] // step
    i = 0 
    temp = []
    for y in range(h_count):
        for x in range(0,w_count):
            x0 = x * step
            x1 = x0 + step
            y0 = y * step
            y1 = y0 + step
#            print(x0,x1,y0,y1)
            patch = img[y0:y1, x0:x1]
            i = i + 1
            img1 = cv2.resize(patch, (224,224), interpolation = cv2.INTER_AREA)
            img1 = img1.reshape(1,224,224,3)
            output = layer_model.predict(img1/255.0)
            temp.append(output[0])
#    print(len(temp))
    data.append(temp)
    data = np.asarray(data)
    return data


'''
获得lstm模型预测的结果矩阵。
舍弃svs图片x和y边缘的512像素。
每个用于预测patch大小为512x512，以这个patch为中心，截取1536x1536大小的图片提取cnn特征，送入lstm模型中进行预测，得到的预测结果为该patch的预测结果。
model：lstm模型,做最后的预测
layer_model：cnn layer model，用于提取cnn特征
svs_file_path：svs图片的保存路径
'''   
def get_out_img(model,layer_model, svs_file_path):   
    print(svs_file_path) 
    step1 = 512
    step = 1536
    livel = 2
    slide = opsl.OpenSlide(svs_file_path)
    Wh = np.zeros((len(slide.level_dimensions),2))
    for i in range (len(slide.level_dimensions)):
        Wh[i,:] = slide.level_dimensions[i]
        Ds = np.zeros((len(slide.level_downsamples),2))
    for i in range (len(slide.level_downsamples)):
        Ds[i,0] = slide.level_downsamples[i]
        Ds[i,1] = slide.get_best_level_for_downsample(Ds[i,0]) 
    w1_count = (int(slide.level_dimensions[0][0]) - 1024) // step1
    h1_count = (int(slide.level_dimensions[0][1]) - 1024) // step1
    
    w_count = int(slide.level_dimensions[0][0]) // step1
    h_count = int(slide.level_dimensions[0][1]) // step1
#    print(w1_count,'\n', h1_count)
    out_img = np.zeros([h_count,w_count])
    out_img1 = np.zeros([h_count,w_count])
    i = 0
    for x in range(w1_count):
        print(x)
        for y in range(h1_count):
            i = i + 1
            svs_data = []
            x0 = step1 + x * step1
#            print('x0:',x0)
            y0 = step1 + y * step1
#            print('y0:',y0)
            x1 = x0 - step1
#            print('x:',x)
            y1 = y0 - step1
#            print('y:',y)
            slide_region1 = np.array(slide.read_region((x0, y0), 0, (step1, step1)))
            slide_img1 = slide_region1[:,:,:3]
            #判断白色像素点
            rgb_s1 = (abs(slide_img1[:,:,0] -107) >= 93) & (abs(slide_img1[:,:,1] -107) >= 93) & (abs(slide_img1[:,:,2] -107) >= 93)
            if np.sum(rgb_s1)<=(step1 * step1 ) * 0.5:
                slide_region = np.array(slide.read_region((x1 , y1  ),0, (step, step)))
                slide_img = slide_region[:,:,:3]
#                print('get')
                svs_data.append(get_imgdata(slide_img, layer_model))
                prob = lstm_model.predict(svs_data)
                preIndex = np.argmax(prob, axis= 1)
                out_img[y,x] = preIndex[0]
                out_img1[y, x] = int(prob[0][3] * 255)

    slide.close()                        
    out_img = cv2.resize(out_img, (int(w_count * step1 /Ds[livel,0]), int(h_count * step1 /Ds[livel,0])), interpolation=cv2.INTER_AREA)
    out_img = cv2.copyMakeBorder(out_img,0,int(Wh[livel,1]-out_img.shape[0]),0,int(Wh[livel,0]-out_img.shape[1]),cv2.BORDER_REPLICATE)
    out_img  = np.uint8(out_img)
    out_img1 = cv2.resize(out_img1, (int(w_count * step1 /Ds[livel,0]), int(h_count * step1 /Ds[livel,0])), interpolation=cv2.INTER_AREA)
    out_img1 = cv2.copyMakeBorder(out_img1,0,int(Wh[livel,1]-out_img1.shape[0]),0,int(Wh[livel,0]-out_img1.shape[1]),cv2.BORDER_REPLICATE)
    out_img1  = np.uint8(out_img1)

    return out_img, out_img1
   
if __name__ == '__main__':  
    model_path = '/cptjack/totem/yatong/4_classes/inceptionResnetV2_0806_3_lstm(0807)/lstm_2.h5'
    save_base_path = '/cptjack/totem/yatong/4_classes/lstm_result/inceptionResnetV2_0806_3_lstm'
    model_path2 = '/cptjack/totem/yatong/4_classes/inceResV2_0806_3/InceptionResnetV2(224).h5'
    model = load_model(model_path2)
    lstm_model = load_model(model_path)
    layer_model = Model(inputs = model.input, outputs = model.layers[-2].output)
    
#    base_data_file = '/cptjack/totem/Data 05272019/Yatong/xml_new_full'
    base_data_file = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/WSI/A_'
    base_data = os.listdir(base_data_file)
    result_map_dir = os.path.sep.join([save_base_path, 'result_map'])
    if not os.path.exists(result_map_dir):os.makedirs(result_map_dir)

    for base_file in base_data:
        if base_file.split('.')[-1] == 'svs':
            l = base_file.split('/')[-1]
            f = l.split('.')[-2]
            print(f)
            name = base_file.split('.')[-2]
            print(base_file)
            xml_name = name + '.xml'        
            svs_file = os.path.sep.join([base_data_file, base_file])
            print(svs_file)
            xml_file = os.path.sep.join([base_data_file, xml_name]) 
            map_name = name + '_32inceReV2_0806_3_lstm_map.png'
            map_path = os.path.sep.join([result_map_dir,map_name])

            iv_map_name = name + 'iv_32inceReV2_0806_3_lstm__map.png'
            iv_map_path = os.path.sep.join([result_map_dir,iv_map_name])
            
            out_img, out_img1 = get_out_img(lstm_model,layer_model,svs_file,name)
            #get_preview函数：获取标注好了的svs二级缩略图   
            pre_img = preview_3.get_preview(svs_file, xml_file)
       
            colormap_dir = os.path.sep.join([save_base_path, 'colormap'])
            colormap_dir2 = os.path.sep.join([save_base_path, 'invasive_colormap'])
            if not os.path.exists(colormap_dir):os.makedirs(colormap_dir)
            if not os.path.exists(colormap_dir2):os.makedirs(colormap_dir2)
            title = name + '_32inceReV2_0806_3_lstm__' + 'colormap'
            colormap.create_colormap(pre_img, out_img, title,  colormap_dir)
            colormap.create_colormap(pre_img, out_img1, title,  colormap_dir2)
               
            cv2.imwrite(map_path, out_img) 
            cv2.imwrite(iv_map_path, out_img1)
          
            del out_img, pre_img, out_img1
            gc.collect()
    
