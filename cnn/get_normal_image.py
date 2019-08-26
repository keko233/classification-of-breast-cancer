# -*- coding: utf-8 -*-
import time
import openslide as opsl
import os
import numpy as np
import cv2
from utils1 import point_polygon_utils as p
#import get_preview_2
import xml.etree.ElementTree as ET
from skimage import io
#from PIL import ImageDraw
#import gc



def get_vertex(vertex_list, level_downsample):
    vertexs = []
    for _, vertex in enumerate(vertex_list):
        x = int(float(vertex['X'])/level_downsample)
        y = int(float(vertex['Y'])/level_downsample)
        vertexs.append((x,y))
    #print(vertexs)
        #vertexs.append((int(float(vertex['X'])/level_downsample), int(float(vertex['Y'])/level_downsample)))
    return vertexs

def get_regions( xml_file):
    try:
        tree = ET.parse(xml_file)
    except:
        return []
    else:
        regions_list = []
        regions_attrib = []
        i = 0
        
        for region in tree.findall('.//Annotation/Regions/Region'):
            vertex_list = []
            regions_attrib.append(region.attrib)
    
            for vertex in region.findall('.//Vertices/Vertex'):
                vertex_list.append(vertex.attrib)
        
#            if (regions_attrib[i]['Type'] == '1'):
            vertexs =get_vertex(vertex_list, 1)
                #vertexs.append(vertexs[0])
            regions_list.append(vertexs)
            i = i + 1  
        
    return regions_list

def position(center,regions):
#    center = (int(x0 + img_w_size/2),int(y0 + img_h_size/2))
#    center =(x, y)
#    if (regions):
    for i in range(len(regions)):
        flag,dis = p.point_position(center,regions[i])
        #print(flag,dis)
        if (flag == True):
            #print(center)
            break
#    else:return False
    return flag
        



#base_data_file = '/cptjack/totem/Data 05272019/Yatong/xml_new_full'  
base_data_file = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/WSI/A_'
#xml_file = '/cptjack/totem/Data 05272019/Yatong/xml_new_full'
pre_heat = '/cptjack/totem/yatong/breast_predict/result/preview_heatmap_result'
#heatmap = '/cptjack/totem/yatong/breast_predict/result/result_heatmap'   
heatmap='/cptjack/totem/yatong/breast_predict/result/FPtrain_Xception_2/result_heatmap'
result_dir = '/cptjack/totem/yatong/4_classes/normal_image/image/'
heatmap_dir = '/cptjack/totem/yatong/4_classes/normal_image/heatmap/'
#pre_heat = os.listdir(pre_heat_file )
#heatmap = os.listdir(heatmap_file)
base_data = os.listdir(base_data_file)
#data = [base_data, pre_heat, heatmap]
for base_file in base_data:
    if base_file.split('.')[-1] == 'svs':
        print(base_file)
        name = base_file.split('.')[-2]
        #svs_file = name + '.svs'
        xml_name = name + '.xml'
#        pre_heat_name = name + '_preHeatResult.jpg'
#        heatmap_name = name + '_heatmap.png'
#        pre_heat_file = os.path.sep.join([pre_heat, pre_heat_name])
#        heatmap_file = os.path.sep.join([heatmap, heatmap_name])
        svs_file_path = os.path.sep.join([base_data_file, base_file])
        xml_file = os.path.sep.join([base_data_file, xml_name])
        #print(base_file, '\n',xml_file, '\n', pre_heat_file, '\n', heatmap_file)
        slide = opsl.OpenSlide(svs_file_path)
#        step = 800
        step_x = 2048
        step_y = 1536
        livel = 2
        level_downsample = slide.level_downsamples[2]
        level_dimension = slide.level_dimensions[2]
        Wh = np.zeros((len(slide.level_dimensions),2))
        for i in range (len(slide.level_dimensions)):
            Wh[i,:] = slide.level_dimensions[i]
            Ds = np.zeros((len(slide.level_downsamples),2))
        for i in range (len(slide.level_downsamples)):
            Ds[i,0] = slide.level_downsamples[i]
            Ds[i,1] = slide.get_best_level_for_downsample(Ds[i,0]) 
        w_count = int(slide.level_dimensions[0][0]) // step_x
        h_count = int(slide.level_dimensions[0][1]) // step_y
        out_img = np.zeros([h_count, w_count])
        regions = get_regions(xml_file)
        i = 0
        for x in range(w_count):
            for y in range(h_count):
                x0 = x * step_x
                y0 = y * step_y
                print(x0, y0)
                slide_region1 = np.array(slide.read_region((x0, y0), 0, (step_x, step_y)))
                slide_img1 = slide_region1[:,:,:3]
                rgb_s1 = (abs(slide_img1[:,:,0] -107) >= 93) & (abs(slide_img1[:,:,1] -107) >= 93) & (abs(slide_img1[:,:,2] -107) >= 93)
                if np.sum(rgb_s1)<=(step_x * step_y ) * 0.5:
                    x1, y1 = x0 + step_x, y0
                    x2, y2 = x0, y0 + step_y
                    x3, y3 = x0 + step_x, y0 + step_y
                    coordinate = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
                    i = i + 1
                    for center in coordinate:
                        flag = position(center,regions)
                        print(flag)
                        if (str(flag) == 'True'):break
                    print(flag)
                    if (str(flag) != 'True'):
                        print(flag, i)
                        out_img[y, x] = 255
                        io.imsave(result_dir + name +'_n_' +str(i) + '.tif', slide_img1)
                        
#                        else:
##                            i = i + 1
#                            print(flag)
#                            out_img[y, x] = 255
#                            io.imsave(result_dir + name +'_n_' +str(i) + '.tif', slide_img1)
        out_img = cv2.resize(out_img, (int(w_count * step_x /Ds[livel,0]), int(h_count * step_y /Ds[livel,0])), interpolation=cv2.INTER_AREA)
        out_img = cv2.copyMakeBorder(out_img,0,int(Wh[livel,1]-out_img.shape[0]),0,int(Wh[livel,0]-out_img.shape[1]),cv2.BORDER_REPLICATE)
        out_img  = np.uint8(out_img)
        cv2.imwrite(heatmap_dir + name +'.png', out_img)
        print(out_img)
        slide.close()
        