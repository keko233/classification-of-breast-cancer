# -*- coding: utf-8 -*-
'''
在有标注的svs大图中截取2048x1536的normal类别大图
截取的标准是该大图的上下左右以及中心坐标都不在标注区域内
'''
import openslide as opsl
import os
import numpy as np
import cv2
from utils1 import point_polygon_utils as p
import xml.etree.ElementTree as ET
from skimage import io

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
获得所需区域的坐标
'''
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

'''
center:坐标点
regions：一个或多个区域的坐标列表
判断坐标是否位于区域中，只要坐标位于一个区域内，就退出循环，直接返回true。
坐标不在所有的区域里面才返回false
'''
def position(center,regions):
    for i in range(len(regions)):
        flag,dis = p.point_position(center,regions[i])
        if (flag == True):
            break
    return flag
        



#base_data_file = '/cptjack/totem/Data 05272019/Yatong/xml_new_full'  
base_data_file = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/WSI/A_'
pre_heat = '/cptjack/totem/yatong/breast_predict/result/preview_heatmap_result' 
heatmap='/cptjack/totem/yatong/breast_predict/result/FPtrain_Xception_2/result_heatmap'
result_dir = '/cptjack/totem/yatong/4_classes/normal_image/image/'
heatmap_dir = '/cptjack/totem/yatong/4_classes/normal_image/heatmap/'
base_data = os.listdir(base_data_file)
for base_file in base_data:
    if base_file.split('.')[-1] == 'svs':
        print(base_file)
        name = base_file.split('.')[-2]
        xml_name = name + '.xml'
        svs_file_path = os.path.sep.join([base_data_file, base_file])
        xml_file = os.path.sep.join([base_data_file, xml_name])
        slide = opsl.OpenSlide(svs_file_path)
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
                    cx, cy = x0 + 1024, y0 + 786
                    coordinate = [(x0, y0), (x1, y1), (x2, y2), (x3, y3), (cx,cy)]
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
                        
        out_img = cv2.resize(out_img, (int(w_count * step_x /Ds[livel,0]), int(h_count * step_y /Ds[livel,0])), interpolation=cv2.INTER_AREA)
        out_img = cv2.copyMakeBorder(out_img,0,int(Wh[livel,1]-out_img.shape[0]),0,int(Wh[livel,0]-out_img.shape[1]),cv2.BORDER_REPLICATE)
        out_img  = np.uint8(out_img)
        cv2.imwrite(heatmap_dir + name +'.png', out_img)
        print(out_img)
        slide.close()
        