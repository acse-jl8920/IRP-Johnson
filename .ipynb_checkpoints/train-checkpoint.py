# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 22:48:30 2021

@author: Johnson
"""

import unet_utils
# import tensorflow as tf
data_path = './crater_data/data_image/*.png'
tar_path = './crater_data/data_targets/*.png'
unet = unet_utils.UNet_Experiment(1,data_path,tar_path,EPOCHS = 10,VERBOSE=(False))
# unet.setup_data_bins(2436, 500, 500)
# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
# a = Image.open('crater_data/data_targets/aeolis_42_5_vh.png')
unet.set_model(0)
unet.compile_test_model()
