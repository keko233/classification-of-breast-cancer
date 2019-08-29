# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os 
import numpy as np
import cv2
import openslide as opsl
from PIL import ImageDraw
import gc


def get_vertex(vertex_list, level_downsample):
    vertexs = []
    for _, vertex in enumerate(vertex_list):
        x = int(float(vertex['X'])/level_downsample)
        y = int(float(vertex['Y'])/level_downsample)
        vertexs.append((x,y))
    #print(vertexs)
        #vertexs.append((int(float(vertex['X'])/level_downsample), int(float(vertex['Y'])/level_downsample)))
    return vertexs


def get_preview(file_path, xml_file):
    slide = opsl.OpenSlide(file_path)
    level_downsample = slide.level_downsamples[2]
    level_dimension = slide.level_dimensions[2]
    thumbnail = slide.get_thumbnail(level_dimension)
    slide.close()
    
    dr = ImageDraw.Draw(thumbnail)
    try:
        tree = ET.parse(xml_file)
    except:
        pass
    else:
        
        regions_attrib = []
        i = 0
        for region in tree.findall('.//Annotation/Regions/Region'):
            vertex_list = []
            regions_attrib.append(region.attrib)
        #print(i, ':', region.attrib, '\n')
        #print(regions_attrib[i]['Type'])
    
            for vertex in region.findall('.//Vertices/Vertex'):
                vertex_list.append(vertex.attrib)
        
            if (regions_attrib[i]['Type'] == '1'):
                vertexs = get_vertex(vertex_list, level_downsample)
                vertexs.append(vertexs[0])
            #print(vertexs)
                dr.line(vertexs, fill = 'red', width = 10)
           
            elif (regions_attrib[i]['Type'] == '2'):
                vertexs = get_vertex(vertex_list, level_downsample)
                dr.arc(vertexs, 0,360,fill = "#ff0000" )
            
            else: 
                vertexs = get_vertex(vertex_list, level_downsample)
                dr.line(vertexs, fill = 'black', width = 10)
            i = i + 1  
       
    return thumbnail


#data_base_path ="/cptjack/totem/Data 05272019/Yatong/xml_new_full"
#save_base_path = "/cptjack/totem/yatong/breast_predict"
###save_base_path = '/cptjack/totem/yatong/breast_predict/result/FPtrain_Xception_2'
#save_preview_dir = os.path.sep.join([save_base_path,'preview_2'])
#if not os.path.exists(save_preview_dir):os.makedirs(save_preview_dir)
#data = os.listdir(data_base_path)
#
#for file in data:
#    if file.split('.')[-1] == 'svs':
#        file_name = file.split('.')[-2]
#        svs_file_path = os.path.sep.join([data_base_path,file])
#        xml_file_name = file_name +'.xml'
#        xml_file_path = os.path.sep.join([data_base_path,xml_file_name])
#        print(xml_file_path)
#       
#        preview_img = get_preview(svs_file_path, xml_file_path)
#        preview = np.asarray(preview_img)
#        preview_name = file_name + '_preview' + '.jpg'
#        preview_save_path = os.path.sep.join([save_preview_dir,preview_name])
#        cv2.imwrite(preview_save_path, preview)
#        del preview_img 
#        gc.collect()



















