# -*- coding: utf-8 -*-
'''
'/cptjack/totem/Data 05272019/Yatong/xml_new_full' 文件夹中的数据对应的画svs图片标注程序
'''
import xml.etree.ElementTree as ET
import openslide as opsl
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

'''
输入：
    vertex_list:xml文件中的坐标
    level_downsample:需要把坐标下采样的倍数。不需要下采样则设为1
输出：
    vertexs：取出的坐标列表。
'''
def get_vertex(vertex_list, level_downsample):
    
    vertexs = []
    for _, vertex in enumerate(vertex_list):
        
        x = int(float(vertex['X'])/level_downsample)
        y = int(float(vertex['Y'])/level_downsample)
        vertexs.append((x,y))
    #print(vertexs)
        #vertexs.append((int(float(vertex['X'])/level_downsample), int(float(vertex['Y'])/level_downsample)))
    return vertexs

'''
svs_file:需要画标注的svs图片保存路径
xml_file:svs图片对应的xml文件保存路径
'''
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
                    dr.line(vertexs, fill = 'red', width = 10)
                else:
                    dr.line(vertexs, fill = 'black', width = 10)
                    
            else:
#                if regions_attrib[i]['Text'] == 'Invasive carcinoma':
                if 'vasive' in regions_attrib[i]['Text']:
#                    print(regions_attrib[i])
                    dr.line(vertexs, fill = 'red', width = 10)
                else:
                    dr.line(vertexs, fill = 'black', width = 10)
                #print(i, ':', regions_attrib, '\n')

            i = i + 1  
        
    return thumbnail
