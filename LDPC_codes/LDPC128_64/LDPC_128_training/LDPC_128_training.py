# -*- coding: utf-8 -*-
import time
T1 = time.time()
import numpy as np
np.set_printoptions(precision=3)
#import matplotlib
import sys
import globalmap as GL
import display_selection as Display
import training_stage as Training_module
   
# Belief propagation using TensorFlow.Run as follows:
    
sys.argv = "python 3.0 3.0 100 100 10 CCSDS_ldpc_n128_k64.alist NMS-1".split()
#sys.argv = "python 2.8 4.0 25 8000 10 wimax_1056_0.83.alist ANMS".split() 
#setting a batch of global parameters
GL.global_setting(sys.argv)    
  
#initial setting for restoring model
restore_info = GL.logistic_setting()
#display the selected decoder
Display.display_selection()
#main part of training
Model = Training_module.training_stage(restore_info)  
Training_module.post_process_input(Model)

T2 =time.time()
print('Running time:%s seconds!'%(T2 - T1))