# -*- coding: utf-8 -*-
from __future__ import division
import sys
sys.path.append('/cptjack/totem/StainTools/')
from utils import visual_utils as vu
from utils import misc_utils as mu
from normalization.reinhard import ReinhardNormalizer
from normalization.macenko import MacenkoNormalizer
from normalization.vahadane import VahadaneNormalizer
from utils1 import preview
from utils1 import get_preview_2 as preview_2
from utils1 import classes4_preview as preview_3
import get_colormap_img as colormap
import openslide as opsl
import numpy as np
import cv2
from PIL import Image,ImageDraw
import os
from keras.models import load_model
import gc
from keras.models import Model
#from utils import judge_position
#from utils import metrics
from skimage import io
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    
def get_out_img(model, svs_file_path,name): 
#    svs_file_path = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Test/WSI/test/test6.svs'
    
    print(svs_file_path) 
#    step1 = 1024
    step1 = 512
#    step1 = 224
  
    print('1')
    livel = 1
    print('1')
    slide = opsl.OpenSlide(svs_file_path)
    print('1')
    Wh = np.zeros((len(slide.level_dimensions),2))
    print('1')
    for i in range (len(slide.level_dimensions)):
        Wh[i,:] = slide.level_dimensions[i]
        Ds = np.zeros((len(slide.level_downsamples),2))
    for i in range (len(slide.level_downsamples)):
        Ds[i,0] = slide.level_downsamples[i]
        Ds[i,1] = slide.get_best_level_for_downsample(Ds[i,0]) 
        
    print(float(Wh[1][0]))
    print(Ds)
    thumb = slide.get_thumbnail(Wh[1])
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
#            print('x0:',x0)
            y0 = y * step1
#            print('y0:',y0)
            slide_region1 = np.array(slide.read_region((x0, y0), 0, (step1, step1)))
            slide_img1 = slide_region1[:,:,:3]
            rgb_s1 = (abs(slide_img1[:,:,0] -107) >= 93) & (abs(slide_img1[:,:,1] -107) >= 93) & (abs(slide_img1[:,:,2] -107) >= 93)
            if np.sum(rgb_s1)<=(step1 * step1 ) * 0.5:
#                print('get')
                img1 = cv2.resize(slide_img1, (224,224), interpolation = cv2.INTER_AREA)
                
                img1 = img1.reshape(1,224,224,3)
                prob = model.predict(img1/255.0)
#                print(prob)
                preIndex = np.argmax(prob, axis= 1)
#                print(preIndex)
                out_img[y,x] = preIndex[0]
                out_img1[y, x] = int(prob[0][3] * 255)
#                out_img[y, x] = prob[0][3]

                             
#    out_img = cv2.resize(out_img, (int(w_count * step1 /Ds[livel,0]), int(h_count * step1 /Ds[livel,0])), interpolation=cv2.INTER_AREA)
#    out_img = cv2.copyMakeBorder(out_img,0,int(Wh[livel,1]-out_img.shape[0]),0,int(Wh[livel,0]-out_img.shape[1]),cv2.BORDER_REPLICATE)
#    out_img  = np.int8(out_img)
#    out_img1 = cv2.resize(out_img1, (int(w_count * step1 /Ds[livel,0]), int(h_count * step1 /Ds[livel,0])), interpolation=cv2.INTER_AREA)
#    out_img1 = cv2.copyMakeBorder(out_img1,0,int(Wh[livel,1]-out_img1.shape[0]),0,int(Wh[livel,0]-out_img1.shape[1]),cv2.BORDER_REPLICATE)
#    out_img1  = np.int8(out_img1)
    slide.close()
    out_img = cv2.resize(out_img, (int(w_count * step1 /Ds[livel,0]), int(h_count * step1 /Ds[livel,0])), interpolation=cv2.INTER_AREA)
    out_img = cv2.copyMakeBorder(out_img,0,int(Wh[livel,1]-out_img.shape[0]),0,int(Wh[livel,0]-out_img.shape[1]),cv2.BORDER_REPLICATE)
    out_img  = np.int8(out_img)
    out_img1 = cv2.resize(out_img1, (int(w_count * step1 /Ds[livel,0]), int(h_count * step1 /Ds[livel,0])), interpolation=cv2.INTER_AREA)
    out_img1 = cv2.copyMakeBorder(out_img1,0,int(Wh[livel,1]-out_img1.shape[0]),0,int(Wh[livel,0]-out_img1.shape[1]),cv2.BORDER_REPLICATE)
    out_img1  = np.int8(out_img1)
    return out_img, out_img1, thumb

if __name__ == '__main__':  
#    model_path = '/cptjack/totem/yatong/4_classes/resnet50_0725/resnet50(224).h5'
#    model_path = '/cptjack/totem/yatong/4_classes/inceResV2_0806_3/InceptionResnetV2(224).h5'
#    model_path = '/cptjack/totem/yatong/4_classes/mil_resnet50_0807/3_resnet50(224).h5'
    model_path = '/cptjack/totem/yatong/4_classes/resnet50_newhsv_0813/resnet50.h5'
#    save_base_path = '/cptjack/totem/yatong/bach_svs_result/cnn_result_2'
#    save_base_path = '/cptjack/totem/yatong/bach_svs_result/InceptionResnetV2_0806_3'
#    save_base_path = '/cptjack/totem/yatong/bach_svs_result/mil_res50(0807)_epoch3'
    save_base_path = '/cptjack/totem/yatong/bach_svs_result/resnet50_newhsv_0813'
    model = load_model(model_path)

    
#    base_data_file = '/cptjack/totem/Data 05272019/Yatong/xml_new_full'
#    base_data_file = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/WSI/A_'
    base_data_file = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Test/WSI/test'
    base_data = os.listdir(base_data_file)
    result_map_dir = os.path.sep.join([save_base_path, 'result_map'])
    if not os.path.exists(result_map_dir):os.makedirs(result_map_dir)


    for base_file in base_data:
        if base_file.split('.')[-1] == 'svs':
            l = base_file.split('/')[-1]
            if 'test1' in l or 'test3' in l or 'test4' in l or 'test6' in l or 'test7' in l or 'test9' in l  or 'test10' in l :continue
            f = l.split('.')[-2]
            print(f)
            name = base_file.split('.')[-2]
            print(base_file)     
            svs_file = os.path.sep.join([base_data_file, base_file])
            print(svs_file)
#            map_name = name + 'resnet50_0725_(0802)_map.png'
#            map_name = name + '_InceptionResnetV2_0806_3_map.png'
#            map_name = name + '_mil_res50(0807)_epoch3_map.png'
            map_name = name + '_resnet50_newhsv_0813_map.png'
            map_path = os.path.sep.join([result_map_dir,map_name])
            
#            iv_map_name = name + 'iv_resnet50_0725(0802)_map.png'
#            iv_map_name = name + '_iv_InceptionResnetV2_0806_3_map.png'
#            iv_map_name = name + '_iv_mil_res50(0807)_epoch3_map.png'
            iv_map_name = name + '_iv_resnet50_newhsv_0813_map.png'
            iv_map_path = os.path.sep.join([result_map_dir,iv_map_name])
            
            out_img, out_img1, thumb = get_out_img(model,svs_file,name)
                
#            pre_img = preview_3.get_preview(svs_file, xml_file)
       
            colormap_dir = os.path.sep.join([save_base_path, 'colormap'])
            colormap_dir2 = os.path.sep.join([save_base_path, 'invasive_colormap'])
            if not os.path.exists(colormap_dir):os.makedirs(colormap_dir)
            if not os.path.exists(colormap_dir2):os.makedirs(colormap_dir2)
#            title = name + 'resnet50_0725_(0802)_' + 'colormap'
#            title = name + '_InceptionResnetV2_0806_3_' + 'colormap'
#            title = name + '_mil_res50(0807)_epoch3_' + 'colormap'
            title = name + '_resnet50_newhsv_0813_colormap'
            colormap.create_colormap(thumb, out_img, title,  colormap_dir)
            colormap.create_colormap(thumb, out_img1, title,  colormap_dir2)
               
            cv2.imwrite(map_path, out_img) 
            cv2.imwrite(iv_map_path, out_img1)
          
            del out_img, out_img1, thumb
            gc.collect()

