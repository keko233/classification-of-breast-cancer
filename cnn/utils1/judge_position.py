# -*- coding: utf-8 -*-

import time
import openslide as opsl
import os
import numpy as np
import cv2
from utils import point_polygon_utils as p
#import get_preview_2
import xml.etree.ElementTree as ET
#from PIL import ImageDraw
#import gc



def get_vertex(vertex_list, level_downsample):
    vertexs = []
    for _, vertex in enumerate(vertex_list):
        #x = int(float(vertex['X'])/level_downsample)
        #y = int(float(vertex['Y'])/level_downsample)
        x = int(float(vertex['X']))
        y = int(float(vertex['Y']))
        vertexs.append((x,y))
    #print(vertexs)
        #vertexs.append((int(float(vertex['X'])/level_downsample), int(float(vertex['Y'])/level_downsample)))
    return vertexs

def get_regions( xml_file, level_downsample):
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
            attribute_list = []
            #print(i, ':', region.attrib, '\n')
            regions_attrib.append(region.attrib)
#           print(len(regions_attrib))

            for vertex in region.findall('.//Vertices/Vertex'):
                vertex_list.append(vertex.attrib)
            vertexs =get_vertex(vertex_list, level_downsample)
            
            if xml_file.split('/')[-1] == 'A01.xml':
                for attribute in region.findall('.//Attributes/Attribute'):
                    attribute_list.append(attribute.attrib)
#                    print(attribute_list)
#                if attribute_list[0]['Value'] == 'Invasive carcinoma':
                if 'Invasive' in attribute_list[0]['Value']:
                    regions_list.append(vertexs)
                    
            else:
#                if regions_attrib[i]['Text'] == 'Invasive carcinoma':
                if 'vasive' in regions_attrib[i]['Text']:
                    regions_list.append(vertexs)
            i = i + 1  
    return regions_list
#        for region in tree.findall('.//Annotation/Regions/Region'):
#            vertex_list = []
#            attribute_list = []
#            regions_attrib.append(region.attrib)
#    
#            for vertex in region.findall('.//Vertices/Vertex'):
#                vertex_list.append(vertex.attrib)
#            vertexs =get_vertex(vertex_list, level_downsample)
#            
#            if xml_file.split('/')[-1] == 'A01.xml':
#                for attribute in region.findall('.//Attributes/Attribute'):
#                    attribute_list.append(attribute.attrib)
##                    print(attribute_list)
##                if attribute_list[0]['Value'] == 'Invasive carcinoma':
#                if 'Invasive' in attribute_list[0]['Value']:
##                    for vertex in region.findall('.//Vertices/Vertex'):
##                        vertex_list.append(vertex.attrib)
##                    vertexs =get_vertex(vertex_list, level_downsample)
#                    
#                    regions_list.append(vertexs)   
#                    
#            else:
##                if regions_attrib[i]['Text'] == 'Invasive carcinoma':
#                if 'vasive' in regions_attrib[i]['Text']:
#                    
##                    for vertex in region.findall('.//Vertices/Vertex'):
##                        vertex_list.append(vertex.attrib)
##                    vertexs =get_vertex(vertex_list, level_downsample)     
#                    regions_list.append(vertexs)
#            i = i + 1     
#        
#    return regions_list

def judge(heatmap_file,w_count, h_count, img_h_size, img_w_size, name):
    img = cv2.imread(heatmap_file)
    i = 0
    w_h_list = []
    #print(w_count * h_count)
    for x in range(1, w_count):
        for y in range(h_count):
            patch = img[int(y * img_h_size):int((y+1)*img_h_size),
                        int(x * img_w_size):int((x+1)*img_w_size)]
            #print(patch.shape)
            
            i = i+1
            file_name =str( i) + '.jpg'
            if(patch[5,5,2] == 255):
                w_h_list.append((x,y))
                #print(i,':',patch[:,:,2])
                #result = '/cptjack/totem/yatong/get_FP_result/img3'
                result = '/cptjack/totem/yatong/breast_predict/result/FPtrain_Xception_2/FP_result/img3'
                result_dir =os.path.sep.join([ result ,name])
                if not os.path.exists(result_dir):os.makedirs(result_dir)
                result_path = os.path.sep.join([result_dir, file_name])
                cv2.imwrite( result_path , patch)
    return w_h_list


    

def position(patch,x0, y0,regions, img_w_size, img_h_size):
    center = (int(x0 + img_w_size/2),int(y0 + img_h_size/2))
    if (regions):
        for i in range(len(regions)):
            flag,dis = p.point_position(center,regions[i])
        #print(flag,dis)
            if (flag == True):
            #print(center)
                break
    else:return False
    return flag
        

#pre_heat_file = '/cptjack/totem/yatong/breast_predict/result/preview_heatmap_result/16545_625969005_preHeatResult.jpg'
#heatmap_file = '/cptjack/totem/yatong/breast_predict/result/result_heatmap/16545_625969005_heatmap.png'   
def get_FP(xml_file, img_h_size, img_w_size, name, heatmap_file,pre_heat_file):
    FP_list = []
    img = cv2.imread(pre_heat_file)
    region = get_regions(xml_file)
    i = 0
    #print(w_count * h_count)
    w_h_list = judge(heatmap_file)
    for w_h in w_h_list:
        y = w_h[1]
        x = w_h[0]
        patch = img[int(y * img_h_size):int((y+1)*img_h_size),
                    int(x * img_w_size):int((x+1)*img_w_size)]
        x0 = x * img_w_size
        y0 = y * img_h_size
        flag = position(patch,x0,y0,region)
        if(flag == True):
            FP_list.append((x,y))
        i = i+1
        file_name =str( i) + '.jpg'
        #result = '/cptjack/totem/yatong/get_FP_result/pre_img'
        result = '/cptjack/totem/yatong/breast_predict/result/FPtrain_Xception_2/FP_result/pre_img'
        result_dir =os.path.sep.join([ result ,name])
        if not os.path.exists(result_dir):os.makedirs(result_dir)
        result_path = os.path.sep.join([result_dir, file_name])
        cv2.imwrite( result_path , patch)    
       
    return FP_list



def get_train(thumbnail, img_h_size, img_w_size, name):
    FP_list= get_FP()
    #print(FP_list)
    filename = 'temp.jpg'
    temp_path = os.path.sep.join([ '/cptjack/totem/yatong/get_FP_result/temp' ,filename])
    thumbnail_img = np.asarray(thumbnail)
    cv2.imwrite(temp_path, thumbnail_img)
    img = cv2.imread(temp_path )
    i = 0
    for x_y in FP_list:
        y = x_y[1]
        x = x_y[0]
        patch = img[int(y * img_h_size):int((y+1)*img_h_size),
                    int(x * img_w_size):int((x+1)*img_w_size)]
        i = i +1
        train_file_name = name + '_'+ str(i) + '_0_.jpg'
        print(train_file_name)
        #train_path = '/cptjack/totem/yatong/get_FP_result/train/0'
        train_path = '/cptjack/totem/yatong/breast_predict/result/FPtrain_Xception_2/train'
        train_dir = os.path.sep.join([train_path,name])
        if not os.path.exists(train_dir):os.makedirs(train_dir)
        result_path = os.path.sep.join([train_dir, train_file_name])
        cv2.imwrite(result_path, patch)
       # gc.collect()
    return