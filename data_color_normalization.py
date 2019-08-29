# -*- coding: utf-8 -*-
'''
染色标准化
'''
from __future__ import division
import sys
sys.path.append('/cptjack/totem/StainTools/')
from utils import visual_utils as vu
from normalization.vahadane import VahadaneNormalizer
from skimage import io   

def get_color_norm(Paths, save_dir):
    #标准图
    n = '/cptjack/totem/StainTools/18655_.tif'
    n_img = vu.read_image(n)
    n2 = VahadaneNormalizer()
    n2.fit(n_img)
    
    for p in Paths:
        f = p.split('/')[-1]
        n = f.split('.')[-2]
        print(n)
        try:
            out= vu.read_image(p)
            img = n2.transform(out)
            io.imsave(save_dir + n +'.tif' ,img)
        except:
            pass

    return       

