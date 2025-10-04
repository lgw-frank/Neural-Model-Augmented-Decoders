# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:09:34 2023

@author: zidonghua_30
"""
import sys
import globalmap as GL
import nn_training as NN_training
import predict_phase as Predict
import nn_net as NN_struct 
#import os

# Set KMP_DUPLICATE_LIB_OK environment variable
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.argv = "python 3.5 3.5 100 8 Hopt_RS_31_25_3_1440ones.alist NMS-1".split()

#setting a batch of global parameters
GL.global_setting(sys.argv) 
original_NMS_indicator = GL.get_map('original_NMS_indicator')
base_ds = GL.base_dataset()
restore_dia_info = GL.logistic_dia_model(original_NMS_indicator)
#determine the priority of order pattern list
if GL.get_map('dia_model_train'):
    DIA_model = NN_struct.conv_bitwise()
    DIA_model = NN_training.Training_dia_model(DIA_model,base_ds,restore_dia_info,original_NMS_indicator) 
    if GL.get_map('ALMLT_available'):
        NN_training.Testing_dia_model(DIA_model,base_ds)
if GL.get_map('swa_model_train') and GL.get_map('ALMLT_available'):
    win_width = GL.get_map('win_width')
    restore_swa_info = GL.logistic_swa_model(original_NMS_indicator)
    restore_list = [restore_dia_info,restore_swa_info]
    DIA_model = NN_struct.conv_bitwise()
    SWA_model = NN_struct.osd_arbitrator(win_width)
    Predict.train_sliding_window(DIA_model, SWA_model,restore_list)
else:
    print('\nTurn on switch of swa_model_train and ensure ALMLT decoding path is ready!')