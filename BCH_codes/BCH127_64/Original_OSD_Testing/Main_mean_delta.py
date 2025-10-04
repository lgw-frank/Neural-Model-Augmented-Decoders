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
import numpy as np
from scipy.special import erf
from scipy.optimize import fsolve


def F_lambda_i(x, sigma):
    if x < 0:
        return 0
    else:
        return 0.5 * (erf((x - 1) / (np.sqrt(2) * sigma)) + erf((x + 1) / (np.sqrt(2) * sigma)))

def inverse_F_lambda_i(y, sigma):
    # Define a function to find the root of
    def equation(x):
        return F_lambda_i(x, sigma) - y
    
    # Initial guess for the root
    x0 = 1.0  # This is a starting point; may need adjustment based on the specific case
    
    # Use fsolve to find the root
    x = fsolve(equation, x0)
    
    return x[0]

def theoretical_ordered_statistics_mean(snr):
    code = GL.get_map('code_parameters')
    sigma =  np.sqrt(1. / (2 * (float(code.k)/float(code.n)) * 10**(snr/10)))
    inverse_list = []
    for i in range(1,code.n+1):
        y_value = i/(code.n+1)
        inverse_value = inverse_F_lambda_i(y_value, sigma)
        #print(f"Inverse F_lambda_i({y_value}) = {inverse_value}")
        inverse_list.append(inverse_value)
    return inverse_list
     
sys.argv = "python 2.0 4.5 6 8 100 BCH_127_64_10_strip.alist NMS-1".split()

#setting a batch of global parameters

GL.global_setting(sys.argv) 
selected_ds_pre,snr_list = GL.data_setting()
selected_ds_pro,_ = GL.post_data_setting()
code = GL.get_map('code_parameters')
x_pos = code.n-code.k   
proportion = 0.7 
FER_list =  []
draw_data = []
for i in range(len(snr_list)):
  snr = round(snr_list[i],1)
  theory_ordered_mean = theoretical_ordered_statistics_mean(snr)
  FER,log_filename,pre_mean_list = NN_test.Testing_OSD(snr,selected_ds_pre[i])
  pro_mean_list = NN_test.Testing_pro_bits(snr,selected_ds_pro[i])
  draw_list =  pre_mean_list+pro_mean_list
  label_list = ['sorted_empirical','swapped_empirical','retested_sorted','retested_swapped']
  delta_draw_list = [draw_list[i]-theory_ordered_mean for i in range(len(draw_list))]
  NN_test.plot_curve(delta_draw_list,label_list,snr,x_pos,proportion)
  FER_list.append((snr,FER))
  draw_data.append((snr,delta_draw_list))
print(f'Summary of FER:{FER_list}')
with open(log_filename,'a+') as f:
  f.write(f'\n Summary of FER:{FER_list}')
  f.write(f'\n Summary of draw_data:{delta_draw_list}')
T2 =time.time()
print('Running time:%s seconds!'%(T2 - T1))