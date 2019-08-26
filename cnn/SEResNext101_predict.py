# -*- coding: utf-8 -*-
from __future__ import division
from utils1 import classes4_preview
#from utils1 import get_preview_2 as preview_2
import get_colormap_img as colormap
import openslide as opsl
import numpy as np
import cv2
from PIL import Image,ImageDraw
import os
import torch
import torch.nn as nn
import gc
import pretrainedmodels
from skimage import io
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2"
device=torch.device('cuda: 0,1,2')
transforms=transforms.ToTensor()

    
def get_out_img(model, svs_file_path,name): 
    
#    print(svs_file_path) 
#    step1 = 1344
    step1 = 224
  
#    print('1')
    livel = 2
#    print('1')
    slide = opsl.OpenSlide(svs_file_path)
#    print('1')
    Wh = np.zeros((len(slide.level_dimensions),2))
#    print('1')
    for i in range (len(slide.level_dimensions)):
        Wh[i,:] = slide.level_dimensions[i]
        Ds = np.zeros((len(slide.level_downsamples),2))
    for i in range (len(slide.level_downsamples)):
        Ds[i,0] = slide.level_downsamples[i]
        Ds[i,1] = slide.get_best_level_for_downsample(Ds[i,0]) 
#    print('1')
#    w1_count = (int(slide.level_dimensions[0][0]) - 1024) // step1
#    h1_count = (int(slide.level_dimensions[0][1]) - 1024) // step1
    
    w_count = int(slide.level_dimensions[0][0]) // step1
    h_count = int(slide.level_dimensions[0][1]) // step1
#    print('1')
    out_img = np.zeros([h_count,w_count])
    i = 0
    for x in range(w_count):
#        print(x)
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
#                try:
#                    slide_img2 = n2.transform(slide_img1)
#             #       io.imshow(slide_img2)
#                except:
#                    slide_img2 = slide_img1
#                    print('did no normalized')

#                print('get img2')
                #img1 = cv2.resize(slide_img1, (224,224), interpolation = cv2.INTER_LINEAR)
#                plt.imshow(slide_img1)
#                plt.show()
                
                img1 = slide_img1.reshape(1,3,224,224)
                img1=torch.tensor(img1,dtype=torch.float32).div(255).cuda(device)
                prob = model(img1)
                result=F.softmax(prob,dim=1).cpu().detach().numpy()
                print(result)
                preIndex = np.argmax(result, axis= 1)
                print(preIndex)
                out_img[y,x] = preIndex[0]
#                out_img[y, x] = int(prob[0][3] * 255)
#                out_img[y, x] = prob[0][3]

                             
    out_img = cv2.resize(out_img, (int(w_count * step1 /Ds[livel,0]), int(h_count * step1 /Ds[livel,0])), interpolation=cv2.INTER_AREA)
    out_img = cv2.copyMakeBorder(out_img,0,int(Wh[livel,1]-out_img.shape[0]),0,int(Wh[livel,0]-out_img.shape[1]),cv2.BORDER_REPLICATE)
    out_img  = np.uint8(out_img)

    return out_img


if __name__ == '__main__':  
    model_name='se_resnext101_32x4d'
    model=pretrainedmodels.__dict__[model_name](num_classes=4,pretrained=None)
    model=nn.DataParallel(model)
#    model.last_linear=nn.Linear(2048,4)
#    model.load_state_dict=torch.load('/cptjack/sys_software_bak/pytorch_models/model_weight/s2340.pth',map_location=device)
#    model.last_linear=nn.Linear(2048,4)
    model.load_state_dict(torch.load('/cptjack/sys_software_bak/pytorch_models/model_weight/se101MIL.pth',map_location=device))
    model.float()
#    if torch.cuda.device_count()>1:
    model.to(device)
#        model=nn.DataParallel(model)    
#    for param in model.parameters():
#        param.requires_grad=False
    model.eval()

    save_base_path = '/cptjack/totem/kaixiangjin/CNNSVS_result/result7'
#    base_data_file = '/cptjack/totem/Data 05272019/Yatong/xml_new_full'
    base_data_file = '/cptjack/totem/kaixiangjin/A_'
    base_data = os.listdir(base_data_file)
    result_map_dir = os.path.sep.join([save_base_path, 'result_map'])
    if not os.path.exists(result_map_dir):os.makedirs(result_map_dir)


    for base_file in base_data:
        if base_file.split('.')[-1] == 'svs':
            l = base_file.split('/')[-1]
            f = l.split('.')[-2]

#            print(f)
            name = base_file.split('.')[-2]
#            print(base_file)
            xml_name = name + '.xml'        
            svs_file = os.path.sep.join([base_data_file, base_file])
#            print(svs_file)
            xml_file = os.path.sep.join([base_data_file, xml_name]) 

            map_name = name + 'se_resnext101_32x4d_cnn(0725)_map.png'
            map_path = os.path.sep.join([result_map_dir,map_name])
    
            out_img = get_out_img(model,svs_file,name)
                
            pre_img = classes4_preview.get_preview(svs_file, xml_file)
       
            colormap_dir = os.path.sep.join([save_base_path, 'colormap'])
            if not os.path.exists(colormap_dir):os.makedirs(colormap_dir)

            title = name + 'se_resnext101_32x4d_cnn(0725)_' + 'colormap'
            colormap.create_colormap(pre_img, out_img, title,  colormap_dir)
               
            cv2.imwrite(map_path, out_img) 
            del out_img, pre_img
            gc.collect()
