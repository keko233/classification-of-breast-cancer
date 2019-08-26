# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import openslide as opsl
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
from imutils import paths
from skimage import io


def get_vertex(vertex_list, level_downsample):
    
    vertexs = []
    for _, vertex in enumerate(vertex_list):
        
        x = int(float(vertex['X'])/level_downsample)
        y = int(float(vertex['Y'])/level_downsample)
#        x = int(float(vertex['X']))
#        y = int(float(vertex['Y']))
        vertexs.append((x,y))
    #print(vertexs)
        #vertexs.append((int(float(vertex['X'])/level_downsample), int(float(vertex['Y'])/level_downsample)))
    return vertexs

def get_preview(svs_file, xml_file):
    slide = opsl.OpenSlide(svs_file)
    level_downsample = slide.level_downsamples[2]
    level_dimension = slide.level_dimensions[2]
    thumbnail = slide.get_thumbnail(level_dimension)
    slide.close()
    
    dr = ImageDraw.Draw(thumbnail)
    
    try:
        tree = ET.parse(xml_file)
    except:
        return []
    else:
#        regions_list = []
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
            
            if svs_file.split('/')[-1] == 'A01.svs':
                for attribute in region.findall('.//Attributes/Attribute'):
                    attribute_list.append(attribute.attrib)
#                    print(attribute_list)
#                if attribute_list[0]['Value'] == 'Invasive carcinoma':
                if 'Invasive' in attribute_list[0]['Value']:
                    dr.line(vertexs, fill = 'red', width = 30)
                elif 'Benign' in attribute_list[0]['Value']:
                    dr.line(vertexs, fill = 'blue', width = 30)
                elif 'situ' in attribute_list[0]['Value']:
                    dr.line(vertexs, fill = 'black', width = 30)
                else:
                    print('get error lable:', attribute_list[0]['Value'])
                    
            else:
#                if regions_attrib[i]['Text'] == 'Invasive carcinoma':
                if 'vasive' in regions_attrib[i]['Text']:
#                    print(regions_attrib[i])
                    dr.line(vertexs, fill = 'red', width = 30)
                elif 'nign' in regions_attrib[i]['Text']:
#                    print(regions_attrib[i])
                    dr.line(vertexs, fill = 'blue', width = 30) 
                elif 'situ' in regions_attrib[i]['Text']:
#                    print(regions_attrib[i])
                    dr.line(vertexs, fill = 'black', width = 30)
                else:
                     print('get error lable:', regions_attrib[i]['Text'])
                #print(i, ':', regions_attrib, '\n')

            i = i + 1  
        
    return thumbnail

#base_data = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/WSI/A_'
#base_data_files = os.listdir(base_data)
##base_data_paths = list(paths.list_images(base_data))
#print(base_data_files)
#result_dir = '/cptjack/totem/yatong/bach_challenge/result/4_classes_preview/'
#for p in base_data_files:
#    n = p.split('.')[-2]
#    if p.split('.')[-1] == 'svs':
#        xml_file = n + '.xml'
#        xml_path = os.path.sep.join([base_data, xml_file])
#        svs_path = os.path.sep.join([base_data, p])
#        preview = get_preview(svs_path, xml_path)
#        io.imsave(result_dir + n + '.png', preview)
#    else:continue