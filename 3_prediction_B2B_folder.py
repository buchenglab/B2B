#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct 13 17:28:37 2026

@author: Guangrui Ding
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import os
import time
import matplotlib.pyplot as plt


from tifffile import imread
from csbdeep.utils import  plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE

model = CARE(config=None, name='my_model_20260831_FP_Beads_fw_2um_axis_0_4unet', basedir='models')
axes = 'ZYX'


datapath = "./data/20260818_FP_Beads_forward_tif_2um"

f = os.listdir(datapath)

save_folder_path = "./results/20260831_FP_Beads_fw"

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

for ii in range(len(f)):
    
    fname = f[ii]
    x = imread(datapath + "/%s" %fname)     
    
    print('image size =', x.shape)
    print('image axes =', axes)

    start = time.time()
    restored = model.predict(x, axes,n_tiles=[8, 8, 8])
    end = time.time()
    elapsed = end - start
    print('Prediction time =', elapsed, ' Seconds')

    save_tiff_imagej_compatible('%s/bkg_%s_%s' % (save_folder_path,model.name,fname), restored, axes )
    save_tiff_imagej_compatible('%s/sub_%s_%s' % (save_folder_path,model.name,fname), x-restored, axes )
    save_tiff_imagej_compatible('%s/devide_%s_%s' % (save_folder_path,model.name,fname), x/restored, axes )

    plt.figure(figsize=(16,10))
    plot_some(np.stack([x,restored]),
              title_list=[['low SNR (maximum projection)','Denoising (maximum projection)']], 
              pmin=2,pmax=99.8);
