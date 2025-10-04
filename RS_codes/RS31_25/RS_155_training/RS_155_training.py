# -*- coding: utf-8 -*-
import time
T1 = time.time()
import numpy as np
np.set_printoptions(precision=3)
#import matplotlib
import sys
import globalmap as GL
import training_stage as Training_module
 
# Belief propagation using TensorFlow.Run as follows:
    
sys.argv = "python 3.5 3.5 100 1000 8 Hopt_RS_31_25_3_1440ones.alist NMS-1".split()
#sys.argv = "python 2.8 4.0 25 8000 10 wimax_1056_0.83.alist ANMS".split() 
#setting a batch of global parameters
GL.global_setting(sys.argv)    

original_NMS_indicator =  GL.get_map('original_NMS_indicator') 

if GL.get_map('enhanced_NMS_indicator') : 
    #initial setting for restoring model
    restore_info = GL.logistic_setting()
    #training for the NMS optimization
    #instance of Model creation   
    NMS_model = Training_module.training_stage(restore_info) 
    Training_module.post_process_input(NMS_model)
    #2nd phase to obtain conventional NMS failure data
if  original_NMS_indicator: 
    #initial setting for restoring model
    restore_info = GL.logistic_setting(original_NMS_indicator)
    #training for the NMS optimization
    #instance of Model creation   
    NMS_model = Training_module.training_stage(restore_info,original_NMS_indicator) 
    Training_module.post_process_input(NMS_model,original_NMS_indicator)
T2 =time.time()
print('Running time:%s seconds!'%(T2 - T1))