# -*- coding: utf-8 -*-
from __future__ import division
from utils1 import classes4_preview as preview_3
import get_colormap_img as colormap
import openslide as opsl
import numpy as np
import cv2
import os
from keras.models import load_model
import gc
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


'''
预测svs大图，得到预测的结果矩阵
输入：
    model：用于预测的模型
    svs_file_path: svs图片的保存路径
    name：svs图片的命名
输出：
    out_img：保存四类别预测结果的矩阵
    out_img:只保存invasive类别（即label=3的类别）预测结果的二值矩阵
'''    
def get_out_img(model, svs_file_path,name): 
    step1 = 512
    livel = 2
    slide = opsl.OpenSlide(svs_file_path)
    Wh = np.zeros((len(slide.level_dimensions),2))
    for i in range (len(slide.level_dimensions)):
        Wh[i,:] = slide.level_dimensions[i]
        Ds = np.zeros((len(slide.level_downsamples),2))
    for i in range (len(slide.level_downsamples)):
        Ds[i,0] = slide.level_downsamples[i]
        Ds[i,1] = slide.get_best_level_for_downsample(Ds[i,0]) 
#    w1_count = (int(slide.level_dimensions[0][0]) - 1024) // step1
#    h1_count = (int(slide.level_dimensions[0][1]) - 1024) // step1
    
    w_count = int(slide.level_dimensions[0][0]) // step1
    h_count = int(slide.level_dimensions[0][1]) // step1
    print('1')
    out_img = np.zeros([h_count,w_count])
    out_img1 = np.zeros([h_count,w_count])
    i = 0
    for x in range(w_count):
        print(x)
        for y in range(h_count):
            i = i + 1          
            x0 =  x * step1
            y0 = y * step1
            slide_region1 = np.array(slide.read_region((x0, y0), 0, (step1, step1)))
            slide_img1 = slide_region1[:,:,:3]
            rgb_s1 = (abs(slide_img1[:,:,0] -107) >= 93) & (abs(slide_img1[:,:,1] -107) >= 93) & (abs(slide_img1[:,:,2] -107) >= 93)
            if np.sum(rgb_s1)<=(step1 * step1 ) * 0.5:
                img1 = cv2.resize(slide_img1, (224,224), interpolation = cv2.INTER_AREA)
                
                img1 = img1.reshape(1,224,224,3)
                prob = model.predict(img1/255.0)
                preIndex = np.argmax(prob, axis= 1)
                out_img[y,x] = preIndex[0]
                out_img1[y, x] = int(prob[0][3] * 255)
#                out_img[y, x] = prob[0][3]

    slide.close()                         
    out_img = cv2.resize(out_img, (int(w_count * step1 /Ds[livel,0]), int(h_count * step1 /Ds[livel,0])), interpolation=cv2.INTER_AREA)
    out_img = cv2.copyMakeBorder(out_img,0,int(Wh[livel,1]-out_img.shape[0]),0,int(Wh[livel,0]-out_img.shape[1]),cv2.BORDER_REPLICATE)
    out_img  = np.uint8(out_img)
    out_img1 = cv2.resize(out_img1, (int(w_count * step1 /Ds[livel,0]), int(h_count * step1 /Ds[livel,0])), interpolation=cv2.INTER_AREA)
    out_img1 = cv2.copyMakeBorder(out_img1,0,int(Wh[livel,1]-out_img1.shape[0]),0,int(Wh[livel,0]-out_img1.shape[1]),cv2.BORDER_REPLICATE)
    out_img1  = np.uint8(out_img1)
    return out_img, out_img1

#####         
if __name__ == '__main__':  
    model_path = '/cptjack/totem/yatong/4_classes/new_mil_resnet50_0813/3_resnet50(224).h5'
    save_base_path = '/cptjack/totem/yatong/4_classes/cnn_result/4_classes_predict_result/mil_resnet50_newhsv_0813/epoch3'   
    model = load_model(model_path)

    
#    base_data_file = '/cptjack/totem/Data 05272019/Yatong/xml_new_full'
    base_data_file = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/WSI/A_'
    base_data = os.listdir(base_data_file)
    result_map_dir = os.path.sep.join([save_base_path, 'result_map'])
    if not os.path.exists(result_map_dir):os.makedirs(result_map_dir)
    iv_result_map_dir = os.path.sep.join([save_base_path, 'iv_result_map'])
    if not os.path.exists(iv_result_map_dir):os.makedirs(iv_result_map_dir)

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
            map_name = name +'_mil_epoch3_resnet50_newhsv_0813_map.png'
            map_path = os.path.sep.join([result_map_dir,map_name])
            
            iv_map_name = name + 'iv_mil_epoch3_resnet50_newhsv_0813_map.png'
            iv_map_path = os.path.sep.join([iv_result_map_dir,iv_map_name])
            
            out_img, out_img1 = get_out_img(model,svs_file,name)
            pre_img = preview_3.get_preview(svs_file, xml_file)
       
            colormap_dir = os.path.sep.join([save_base_path, 'colormap'])
            colormap_dir2 = os.path.sep.join([save_base_path, 'invasive_colormap'])
            if not os.path.exists(colormap_dir):os.makedirs(colormap_dir)

            title = name + '_mil_epoch3_resnet50_newhsv_0813' +'_colormap'
            colormap.create_colormap(pre_img, out_img, title,  colormap_dir)
            colormap.create_colormap(pre_img, out_img1, title,  colormap_dir2)
               
            cv2.imwrite(map_path, out_img) 
            cv2.imwrite(iv_map_path, out_img1)
            del out_img, pre_img, out_img1
            gc.collect()
