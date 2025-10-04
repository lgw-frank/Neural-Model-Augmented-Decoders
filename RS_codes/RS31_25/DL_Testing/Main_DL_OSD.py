# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:09:34 2023

@author: zidonghua_30
"""
import time
T1 = time.time()
import sys
import globalmap as GL
import nn_testing as NN_test
import ordered_statistics_decoding as OSD_mod

sys.argv = "python 2.0 4.5 6 100  8 Hopt_RS_31_25_3_1440ones.alist NMS-1".split()

#setting a batch of global parameters

GL.global_setting(sys.argv) 
selected_ds,snr_list = GL.data_setting()

soft_margin = GL.get_map('soft_margin')
code = GL.get_map('code_parameters')

logistics = GL.logisticis_setting()

DIA_model,SWA_model = NN_test.NNs_gen(logistics[0]) 
model_list = [DIA_model,SWA_model]
OSD_instance = OSD_mod.osd(code)
FER_list =  []
for i in range(len(snr_list)):
  snr = round(snr_list[i],2)
  FER,log_filename = NN_test.Testing_OSD(OSD_instance,snr,selected_ds[i],model_list,logistics)
  FER_list.append((snr,FER))
if GL.get_map('sliding_strategy'):
    sub_title = f"soft_margin={soft_margin}"
else:
    sub_title = "max_decoding"
print(f'Summary of FER under {sub_title}:\n{FER_list} ')
with open(log_filename,'a+') as f:
  f.write(f'\nSummary of FER under {sub_title}:\n{FER_list}')
T2 = time.time()
print('Running time:%s seconds!'%(T2 - T1))